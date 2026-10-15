"""
utils.py — Shared utility functions used across all notebooks.

Includes:
  - Training loop helpers (train_one_epoch, validate_one_epoch)
  - Checkpoint save / load
  - Sliding window inference for full 3D volumes
  - Learning rate scheduler setup
  - TensorBoard logging helpers
  - Seed fixing
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.config import CHECKPOINT_DIR, LOGS_DIR, PATCH_SIZE, PATCH_OVERLAP, DEVICE, USE_AMP


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def save_checkpoint(model: nn.Module, optimizer, epoch: int,
                    val_dice: float, model_name: str,
                    is_best: bool = False):
    """
    Save model checkpoint. Always saves latest; also saves best separately.
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    state = {
        "epoch":      epoch,
        "model_name": model_name,
        "val_dice":   val_dice,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    # Latest checkpoint (overwrites every epoch)
    latest_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_latest.pth")
    torch.save(state, latest_path)

    # Best checkpoint
    if is_best:
        best_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pth")
        torch.save(state, best_path)
        print(f"  ✅ New best! Saved: {best_path}  (val_dice={val_dice:.4f})")

    return latest_path


def load_checkpoint(model: nn.Module, optimizer=None,
                    model_name: str = "model", prefer_best: bool = True
                    ) -> Tuple[int, float]:
    """
    Load checkpoint if it exists. Returns (start_epoch, best_val_dice).
    If no checkpoint found, returns (0, 0.0) so training starts fresh.
    """
    if prefer_best:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_best.pth")
    else:
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_latest.pth")

    if not os.path.exists(ckpt_path):
        print(f"  No checkpoint found at {ckpt_path} — starting fresh.")
        return 0, 0.0

    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    epoch    = state.get("epoch", 0)
    val_dice = state.get("val_dice", 0.0)
    print(f"  ✅ Resumed from: {ckpt_path}  (epoch={epoch}, val_dice={val_dice:.4f})")
    return epoch + 1, val_dice


# ---------------------------------------------------------------------------
# Training & validation loops
# ---------------------------------------------------------------------------

def train_one_epoch(model: nn.Module, loader, optimizer,
                    criterion, device, epoch: int,
                    scaler=None,
                    writer: Optional[SummaryWriter] = None) -> Dict[str, float]:
    """
    One full pass over the training set.
    Uses BF16 AMP if USE_AMP=True in config (significant speedup on B200 MIG).
    Returns dict with average losses for the epoch.
    """
    model.train()
    total_loss = dice_loss_sum = focal_loss_sum = 0.0
    n_batches = 0

    # Defensively handle case where writer was passed positionally as 7th argument
    if scaler is not None and (isinstance(scaler, SummaryWriter) or hasattr(scaler, "add_scalar")):
        writer = scaler
        scaler = None

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for images, targets in pbar:
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        if USE_AMP and device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(images)
                loss, dl, fl = criterion(logits, targets)
            if scaler is not None and hasattr(scaler, "scale"):
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        else:
            logits = model(images)
            loss, dl, fl = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss     += loss.item()
        dice_loss_sum  += dl.item()
        focal_loss_sum += fl.item()
        n_batches      += 1

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    metrics = {
        "loss":       total_loss / n_batches,
        "dice_loss":  dice_loss_sum / n_batches,
        "focal_loss": focal_loss_sum / n_batches,
    }

    if writer:
        for k, v in metrics.items():
            writer.add_scalar(f"Train/{k}", v, epoch)

    return metrics


@torch.no_grad()
def validate_one_epoch(model: nn.Module, loader, criterion,
                       device, epoch: int,
                       writer: Optional[SummaryWriter] = None) -> Dict[str, float]:
    """
    One full pass over the validation set (no grad).
    Returns dict with average loss and mean Dice score.
    """
    from src.metrics import compute_patient_metrics, dice_score, get_regions

    model.eval()
    total_loss = 0.0
    all_wt_dice = []
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]", leave=False)
    for images, targets in pbar:
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if USE_AMP and device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(images)
                loss, _, _ = criterion(logits, targets)
        else:
            logits = model(images)
            loss, _, _ = criterion(logits, targets)
        total_loss += loss.item()
        n_batches  += 1

        # Compute WT Dice for monitoring
        preds = logits.argmax(dim=1).cpu().numpy()  # (B, H, W, D)
        segs  = targets.cpu().numpy()
        for p, s in zip(preds, segs):
            p_wt = (p > 0)
            s_wt = (s > 0)
            all_wt_dice.append(dice_score(p_wt, s_wt))

    mean_wt_dice = float(np.mean(all_wt_dice)) if all_wt_dice else 0.0
    metrics = {
        "loss":    total_loss / max(n_batches, 1),
        "wt_dice": mean_wt_dice,
    }

    if writer:
        for k, v in metrics.items():
            writer.add_scalar(f"Val/{k}", v, epoch)

    return metrics


# ---------------------------------------------------------------------------
# Sliding window inference (for full-volume prediction)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sliding_window_inference(model: nn.Module, volume: torch.Tensor,
                              patch_size: tuple = PATCH_SIZE,
                              overlap: tuple = PATCH_OVERLAP,
                              device=DEVICE) -> np.ndarray:
    """
    Run model on a full 3D volume using overlapping patches.
    Averages predictions in overlapping regions for smooth boundaries.

    Args:
        model   : trained model (eval mode)
        volume  : (1, 4, H, W, D) or (4, H, W, D) tensor
        patch_size : (ph, pw, pd)
        overlap    : (oh, ow, od) — how much adjacent patches overlap

    Returns:
        pred_seg : (H, W, D) integer array — class labels
    """
    model.eval()
    if volume.dim() == 4:
        volume = volume.unsqueeze(0)  # add batch dim

    _, C, H, W, D = volume.shape
    ph, pw, pd = patch_size
    oh, ow, od = overlap

    # Accumulators for predictions
    pred_sum   = torch.zeros(1, 4, H, W, D)  # sum of softmax probs
    count_map  = torch.zeros(1, 1, H, W, D)  # how many patches covered each voxel

    # Stride = patch_size - overlap
    sh, sw, sd = ph - oh, pw - ow, pd - od

    for zi in range(0, max(1, D - pd + 1), sd):
        for yi in range(0, max(1, H - ph + 1), sh):
            for xi in range(0, max(1, W - pw + 1), sw):
                ze = min(zi + pd, D)
                ye = min(yi + ph, H)
                xe = min(xi + pw, W)

                # Extract patch
                patch = volume[:, :, yi:ye, xi:xe, zi:ze].to(device)

                # Pad if patch is smaller than patch_size at edges
                pad_h = ph - (ye - yi)
                pad_w = pw - (xe - xi)
                pad_d = pd - (ze - zi)
                if pad_h > 0 or pad_w > 0 or pad_d > 0:
                    patch = F.pad(patch, (0, pad_d, 0, pad_w, 0, pad_h))

                logits = model(patch)                           # (1, 4, ph, pw, pd)
                probs  = F.softmax(logits, dim=1).cpu()

                # Trim back if padded
                probs = probs[:, :, :ye-yi, :xe-xi, :ze-zi]

                pred_sum[:, :, yi:ye, xi:xe, zi:ze] += probs
                count_map[:, :, yi:ye, xi:xe, zi:ze] += 1

    # Average and argmax
    pred_avg = pred_sum / count_map.clamp(min=1)
    pred_seg = pred_avg.argmax(dim=1).squeeze(0).numpy().astype(np.int64)

    return pred_seg


# ---------------------------------------------------------------------------
# LR Scheduler
# ---------------------------------------------------------------------------

def get_scheduler(optimizer, patience: int = 10, factor: float = 0.5):
    return ReduceLROnPlateau(
        optimizer, mode="max", patience=patience,
        factor=factor, min_lr=1e-6
    )


# ---------------------------------------------------------------------------
# TensorBoard writer factory
# ---------------------------------------------------------------------------

def get_writer(model_name: str) -> SummaryWriter:
    log_dir = os.path.join(LOGS_DIR, model_name)
    os.makedirs(log_dir, exist_ok=True)
    return SummaryWriter(log_dir)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"
