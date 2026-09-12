"""
losses.py — Loss Functions for Multi-Class 3D Brain Tumor Segmentation
Implements:
  - Multi-Class Soft Dice Loss (averaged over foreground classes 1, 2, 3)
  - Weighted Cross-Entropy Loss (with training-set-derived class weights)
  - Multi-Class Focal Loss
  - Combined Loss (Dice + Cross-Entropy / Focal)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# =============================================================================
# 1. Multi-Class Soft Dice Loss
# =============================================================================

class DiceLoss(nn.Module):
    """
    Soft Multi-Class Dice Loss.
    Calculates Dice similarity per class and averages over foreground classes (1, 2, 3)
    to prevent healthy background voxels (98% of scan) from dominating the objective.

    Args:
        num_classes : Total classes including background (4 for BraTS)
        smooth      : Laplace smoothing constant to prevent division by zero
        ignore_bg   : If True, excludes class 0 (background) from the average
    """

    def __init__(self, num_classes: int = 4, smooth: float = 1e-5, ignore_bg: bool = True):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_bg = ignore_bg

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, C, H, W, D) — raw unnormalized network predictions
            targets : (B, H, W, D)    — integer class labels in {0, 1, 2, 3}
        Returns:
            scalar Dice loss in [0, 1]
        """
        probs = F.softmax(logits, dim=1)  # (B, C, H, W, D)

        # One-hot encode ground truth: (B, H, W, D) -> (B, H, W, D, C) -> (B, C, H, W, D)
        targets_oh = F.one_hot(targets.clamp(0, self.num_classes - 1), self.num_classes)
        targets_oh = targets_oh.permute(0, 4, 1, 2, 3).float()

        # Sum over spatial and batch dimensions
        spatial_dims = (0, 2, 3, 4)
        intersection = (probs * targets_oh).sum(dim=spatial_dims)
        cardinality  = (probs + targets_oh).sum(dim=spatial_dims)

        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        start_class = 1 if self.ignore_bg else 0
        loss = 1.0 - dice_per_class[start_class:].mean()
        return loss


# =============================================================================
# 2. Multi-Class Focal Loss
# =============================================================================

class FocalLoss(nn.Module):
    """
    Multi-Class Focal Loss — down-weights well-classified easy voxels (e.g. background)
    and concentrates learning on hard, ambiguous tumor boundaries.

    Args:
        gamma : Focusing parameter (gamma=2.0 standard)
        alpha : Optional 1D tensor of class weights (C,) derived from training set
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, C, H, W, D)
            targets : (B, H, W, D)
        """
        B, C, H, W, D = logits.shape
        logits_flat  = logits.permute(0, 2, 3, 4, 1).reshape(-1, C)  # (N, C)
        targets_flat = targets.reshape(-1)                             # (N,)

        log_probs = F.log_softmax(logits_flat, dim=1)
        probs     = torch.exp(log_probs)

        # Log prob and prob of true classes
        log_pt = log_probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
        pt     = probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)

        focal_weight = (1.0 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_tensor = self.alpha.to(logits.device)
            alpha_t = alpha_tensor[targets_flat]
            focal_weight = focal_weight * alpha_t

        loss = -(focal_weight * log_pt).mean()
        return loss


# =============================================================================
# 3. Combined Loss (Dice + Cross-Entropy / Focal)
# =============================================================================

class CombinedLoss(nn.Module):
    """
    Weighted combination of Soft Dice Loss + Cross-Entropy Loss (or Focal Loss).
    Guarantees both global volumetric overlap optimization and sharp boundary gradients.

    Args:
        num_classes  : 4 (Background, NCR, ED, ET)
        dice_weight  : Weight for Dice loss component (default: 0.5)
        ce_weight    : Weight for CE/Focal loss component (default: 0.5)
        use_focal    : If True, uses FocalLoss; if False, uses standard CrossEntropyLoss
        focal_gamma  : Focusing exponent for FocalLoss
        class_weights: Optional class weight tensor (C,) computed strictly from training split
    """

    def __init__(self,
                 num_classes: int = 4,
                 dice_weight: float = 0.5,
                 ce_weight: float = 0.5,
                 use_focal: bool = True,
                 focal_gamma: float = 2.0,
                 class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss(num_classes=num_classes, ignore_bg=True)

        self.use_focal = use_focal
        if use_focal:
            self.secondary_loss = FocalLoss(gamma=focal_gamma, alpha=class_weights)
        else:
            self.secondary_loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            total_loss, dice_loss, secondary_loss (all scalar tensors)
        """
        d_loss = self.dice_loss(logits, targets)
        s_loss = self.secondary_loss(logits, targets)
        total = self.dice_weight * d_loss + self.ce_weight * s_loss
        return total, d_loss, s_loss


# =============================================================================
# 4. Training-Set-Only Class Weight Estimator
# =============================================================================

def compute_class_weights_from_train_split(train_loader, num_classes: int = 4, max_batches: int = 50) -> torch.Tensor:
    """
    Computes inverse frequency class weights STRICTLY from the training set dataloader,
    preventing any test-set statistic leakage.
    """
    counts = torch.zeros(num_classes, dtype=torch.float64)
    batches = 0
    for _, targets in train_loader:
        for c in range(num_classes):
            counts[c] += (targets == c).sum().item()
        batches += 1
        if batches >= max_batches:
            break

    total_voxels = counts.sum()
    if total_voxels == 0:
        return torch.ones(num_classes, dtype=torch.float32)

    # Inverse class frequency with median-based scaling
    weights = total_voxels / (num_classes * counts.clamp(min=1.0))
    weights = weights / weights.sum() * num_classes
    return weights.float()
