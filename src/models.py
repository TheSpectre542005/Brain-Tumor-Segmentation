"""
models.py — Unified 3D Brain Tumor Segmentation Architectures
Implements:
  1. Baseline 3D U-Net (Çiçek et al., 2016)
  2. Channel Attention U-Net (Squeeze-and-Excitation, Hu et al., 2018)
  3. Spatial Attention U-Net (Spatial Attention Gates, Oktay et al., 2018)
  4. Proposed Dual-Attention 3D U-Net (Hybrid SE + Spatial Attention)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Union


# =============================================================================
# 1. Attention Building Blocks
# =============================================================================

class SEBlock3D(nn.Module):
    """
    3D Squeeze-and-Excitation (SE) Block — Channel Attention.
    Learns dynamic inter-channel dependencies to answer: "WHICH feature channels are informative?"

    Mechanism:
      ① Squeeze : AdaptiveAvgPool3D(1) → (B, C, 1, 1, 1)
      ② Excite  : Linear(C -> C//r) -> ReLU -> Linear(C//r -> C) -> Sigmoid → (B, C, 1, 1, 1)
      ③ Scale   : Voxel-wise channel multiplication X * S
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        bottleneck = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(channels, bottleneck, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _, _ = x.shape
        scale = self.fc(x).view(B, C, 1, 1, 1)
        return x * scale


class SpatialAttentionGate3D(nn.Module):
    """
    3D Spatial Attention Gate (AG) — Spatial Attention on Skip Connections.
    Uses deep, coarse decoder gating signal (g) to suppress irrelevant background
    regions in shallow encoder skip connections (x).

    Answers: "WHERE in the 3D volume is the tumor tissue located?"
    """

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            g : (B, F_g, H_g, W_g, D_g) — coarse gating signal from deeper decoder
            x : (B, F_l, H, W, D)       — skip connection feature from encoder
        Returns:
            attended_x : (B, F_l, H, W, D) — skip connection reweighted by spatial attention map
            alpha      : (B, 1, H, W, D)   — 3D spatial attention coefficient map in [0, 1]
        """
        # Project gating signal and upsample to match skip connection spatial dimensions
        g1 = self.W_g(g)
        if g1.shape[2:] != x.shape[2:]:
            g1 = F.interpolate(g1, size=x.shape[2:], mode="trilinear", align_corners=True)

        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        alpha = self.psi(psi)  # (B, 1, H, W, D)

        attended_x = x * alpha
        return attended_x, alpha


# =============================================================================
# 2. Convolutional Units & Encoder / Decoder Blocks
# =============================================================================

class DoubleConv3D(nn.Module):
    """
    Standard Two-Stage 3D Convolutional Unit:
      [Conv3D (3x3x3) -> BatchNorm3D -> ReLU] x 2
      Optionally followed by an SE Channel Attention block.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 mid_channels: Optional[int] = None,
                 use_se: bool = False, se_reduction: int = 16):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.se = SEBlock3D(out_channels, se_reduction) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.conv(x))


class EncoderBlock(nn.Module):
    """Downsampling stage: MaxPool3D (2x2x2) -> DoubleConv3D (with optional SE)."""

    def __init__(self, in_channels: int, out_channels: int, use_se: bool = False):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv = DoubleConv3D(in_channels, out_channels, use_se=use_se)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class AttentionDecoderBlock(nn.Module):
    """
    Upsampling stage with optional Spatial Attention Gate and Channel Attention:
      1. Upsample feature map via ConvTranspose3D (2x2x2)
      2. Apply Spatial Attention Gate to encoder skip connection (if enabled)
      3. Concatenate upsampled feature with (attended) skip feature
      4. DoubleConv3D (with optional SE)
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 use_spatial_attn: bool = True, use_se: bool = True):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.use_spatial_attn = use_spatial_attn
        if use_spatial_attn:
            self.attn_gate = SpatialAttentionGate3D(
                F_g=in_channels // 2,
                F_l=skip_channels,
                F_int=max(skip_channels // 2, 4)
            )
        else:
            self.attn_gate = None

        self.conv = DoubleConv3D(in_channels // 2 + skip_channels, out_channels, use_se=use_se)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = self.up(x)

        # Handle any dimension oddities gracefully
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=True)

        attn_map = None
        if self.attn_gate is not None:
            skip, attn_map = self.attn_gate(g=x, x=skip)

        x_cat = torch.cat([skip, x], dim=1)
        out = self.conv(x_cat)
        return out, attn_map


# =============================================================================
# 3. Unified 3D Architecture & Ablation Model Class
# =============================================================================

class AttentionUNet3D(nn.Module):
    """
    Unified 3D Attention-Augmented U-Net supporting all ablation configurations:
      - Baseline 3D U-Net        : use_channel_attn=False, use_spatial_attn=False
      - Channel Attention Only   : use_channel_attn=True,  use_spatial_attn=False
      - Spatial Attention Only   : use_channel_attn=False, use_spatial_attn=True
      - Hybrid / Full Attention  : use_channel_attn=True,  use_spatial_attn=True

    Args:
        in_channels      : Number of input MRI sequences (4 for BraTS: FLAIR, T1, T1ce, T2)
        out_channels     : Number of segmentation classes (4: Background, NCR, ED, ET)
        init_features    : Number of base filters in encoder level 1 (default: 32)
        use_channel_attn : Enable Squeeze-and-Excitation channel attention in conv blocks
        use_spatial_attn : Enable Spatial Attention Gates on decoder skip connections
    """

    def __init__(self,
                 in_channels: int = 4,
                 out_channels: int = 4,
                 init_features: int = 32,
                 use_channel_attn: bool = True,
                 use_spatial_attn: bool = True):
        super().__init__()
        f = init_features
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_channel_attn = use_channel_attn
        self.use_spatial_attn = use_spatial_attn

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc1 = DoubleConv3D(in_channels, f, use_se=use_channel_attn)        # f (32)
        self.enc2 = EncoderBlock(f, f * 2, use_se=use_channel_attn)              # f*2 (64)
        self.enc3 = EncoderBlock(f * 2, f * 4, use_se=use_channel_attn)          # f*4 (128)
        self.enc4 = EncoderBlock(f * 4, f * 8, use_se=use_channel_attn)          # f*8 (256)

        # ── Bottleneck ───────────────────────────────────────────────────────
        self.bottleneck = EncoderBlock(f * 8, f * 16, use_se=use_channel_attn)   # f*16 (512)

        # ── Decoder with Skip Connections ────────────────────────────────────
        self.dec4 = AttentionDecoderBlock(f * 16, f * 8, f * 8, use_spatial_attn, use_channel_attn)
        self.dec3 = AttentionDecoderBlock(f * 8, f * 4, f * 4, use_spatial_attn, use_channel_attn)
        self.dec2 = AttentionDecoderBlock(f * 4, f * 2, f * 2, use_spatial_attn, use_channel_attn)
        self.dec1 = AttentionDecoderBlock(f * 2, f, f, use_spatial_attn, use_channel_attn)

        # ── Segmentation Head ────────────────────────────────────────────────
        self.head = nn.Conv3d(f, out_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass for training / inference.
        Args:
            x : (B, 4, H, W, D) input MRI patch or volume
        Returns:
            logits : (B, 4, H, W, D) unnormalized class logits
        """
        # Encoder
        s1 = self.enc1(x)         # (B, f,    H,   W,   D)
        s2 = self.enc2(s1)        # (B, f*2,  H/2, W/2, D/2)
        s3 = self.enc3(s2)        # (B, f*4,  H/4, W/4, D/4)
        s4 = self.enc4(s3)        # (B, f*8,  H/8, W/8, D/8)

        # Bottleneck
        b = self.bottleneck(s4)   # (B, f*16, H/16,W/16,D/16)

        # Decoder
        d4, _ = self.dec4(b, s4)  # (B, f*8,  H/8, W/8, D/8)
        d3, _ = self.dec3(d4, s3) # (B, f*4,  H/4, W/4, D/4)
        d2, _ = self.dec2(d3, s2) # (B, f*2,  H/2, W/2, D/2)
        d1, _ = self.dec1(d2, s1) # (B, f,    H,   W,   D)

        return self.head(d1)

    def get_attention_maps(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass returning both prediction logits and 3D spatial attention maps
        from all 4 decoder levels (for clinical interpretability and visualization).
        """
        # Encoder
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        # Bottleneck
        b = self.bottleneck(s4)

        # Decoder with attention map recording
        attention_maps = []
        d4, a4 = self.dec4(b, s4)
        if a4 is not None:
            attention_maps.append(a4.detach().cpu())

        d3, a3 = self.dec3(d4, s3)
        if a3 is not None:
            attention_maps.append(a3.detach().cpu())

        d2, a2 = self.dec2(d3, s2)
        if a2 is not None:
            attention_maps.append(a2.detach().cpu())

        d1, a1 = self.dec1(d2, s1)
        if a1 is not None:
            attention_maps.append(a1.detach().cpu())

        logits = self.head(d1)
        return logits, attention_maps


class UNet3D(AttentionUNet3D):
    """Baseline 3D U-Net without attention (Çiçek et al. 2016)."""

    def __init__(self, in_channels: int = 4, out_channels: int = 4, init_features: int = 32):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            init_features=init_features,
            use_channel_attn=False,
            use_spatial_attn=False
        )


# =============================================================================
# 4. Model Factory & Utilities
# =============================================================================

def build_model(variant: str = "hybrid",
                in_channels: int = 4,
                out_channels: int = 4,
                init_features: int = 32) -> AttentionUNet3D:
    """
    Factory function instantiating model variants for training and ablation experiments.

    Variants:
      - 'baseline'                 : Standard 3D U-Net (no attention)
      - 'channel'                  : Squeeze-and-Excitation channel attention only
      - 'spatial'                  : Spatial Attention Gates only
      - 'hybrid' / 'full' / 'ours' : Dual-Attention 3D U-Net (SE + Spatial Attention)
    """
    variant = variant.lower().strip()
    if variant in ["baseline", "unet3d", "plain"]:
        return AttentionUNet3D(in_channels, out_channels, init_features,
                               use_channel_attn=False, use_spatial_attn=False)
    elif variant in ["channel", "se", "channel_only", "attention_channel"]:
        return AttentionUNet3D(in_channels, out_channels, init_features,
                               use_channel_attn=True, use_spatial_attn=False)
    elif variant in ["spatial", "gate", "spatial_only", "attention_spatial"]:
        return AttentionUNet3D(in_channels, out_channels, init_features,
                               use_channel_attn=False, use_spatial_attn=True)
    elif variant in ["hybrid", "full", "dual", "ours", "attention_unet", "attention_full"]:
        return AttentionUNet3D(in_channels, out_channels, init_features,
                               use_channel_attn=True, use_spatial_attn=True)
    else:
        raise ValueError(f"Unknown model variant: '{variant}'. Choose from: baseline, channel, spatial, hybrid.")


def count_parameters(model: nn.Module) -> Dict[str, Union[int, str]]:
    """Returns parameter count dict and formatted string."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "formatted": f"Total: {total:,} | Trainable: {trainable:,}"
    }


if __name__ == "__main__":
    # Sanity check forward pass for all 4 variants
    variants = ["baseline", "channel", "spatial", "hybrid"]
    x = torch.randn(1, 4, 96, 96, 96)
    print("Testing 3D Neural Network Architectures:")
    for v in variants:
        m = build_model(v)
        out = m(x)
        info = count_parameters(m)
        print(f"  [{v.upper():8s}] {info['formatted']} | Output shape: {tuple(out.shape)}")
