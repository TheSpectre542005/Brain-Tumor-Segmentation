"""
utils.py — Shared Utilities for Training, Resumption, Sliding Window Inference & Logging
"""

import os
import random
from pathlib import Path
from typing import Optional, Dict, Tuple, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.config import (
    CHECKPOINT_DIR, LOGS_DIR, PATCH_SIZE, PATCH_OVERLAP,
    RANDOM_SEED, NUM_CLASSES, get_device
)


# =============================================================================
# 1. Reproducibility
# =============================================================================

def set_seed(seed: int = RANDOM_SEED) -> None:
    """Sets random seeds across Python, NumPy, PyTorch CPU & CUDA for strict reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# 2. Resumable Checkpoint Management
# =============================================================================

def save_checkpoint(model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                    scaler: Optional[torch.cuda.amp.GradScaler],
                    epoch: int,
                    val_metric: float,
                    model_name: str,
                    checkpoint_dir: str = CHECKPOINT_DIR,
                    is_best: bool = False,
                    extra_info: Optional[Dict] = None) -> str:
    """
    Saves comprehensive checkpoint state for exact crash resumption.
    Saves both latest checkpoint and best checkpoint.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    state = {
        "epoch":                epoch,
        "model_name":           model_name,
        "val_metric":           val_metric,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict":    scaler.state_dict() if scaler is not None else None,
        "torch_rng_state":      torch.get_rng_state(),
        "numpy_rng_state":      np.random.get_state(),
        "extra_info":           extra_info or {},
    }
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state_all()

    latest_path = os.path.join(checkpoint_dir, f"{model_name}_latest.pth")
    torch.save(state, latest_path)

    if is_best:
        best_path = os.path.join(checkpoint_dir, f"{model_name}_best.pth")
        torch.save(state, best_path)

    return latest_path


def load_checkpoint(model: nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                    scaler: Optional[torch.cuda.amp.GradScaler] = None,
                    checkpoint_path: Optional[str] = None,
                    model_name: str = "model",
                    checkpoint_dir: str = CHECKPOINT_DIR,
                    prefer_best: bool = False,
                    device: Optional[torch.device] = None
                    ) -> Tuple[int, float]:
    """
    Restores full training state from a saved checkpoint file.

    Returns:
        (start_epoch, best_val_metric)
    """
    if device is None:
        device = get_device()

    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        target_name = f"{model_name}_best.pth" if prefer_best else f"{model_name}_latest.pth"
        checkpoint_path = os.path.join(checkpoint_dir, target_name)

    if not os.path.exists(checkpoint_path):
        # Fallback to check if best exists if latest doesn't, or vice-versa
        alt_name = f"{model_name}_latest.pth" if prefer_best else f"{model_name}_best.pth"
        alt_path = os.path.join(checkpoint_dir, alt_name)
        if os.path.exists(alt_path):
            checkpoint_path = alt_path
        else:
            print(f"  [INFO] No checkpoint found at {checkpoint_path} — starting fresh.")
            return 0, 0.0

    print(f"  [CHECKPOINT] Loading state from: {checkpoint_path}")
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=device)

    # 1. Model weights
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    elif isinstance(state, dict):
        model.load_state_dict(state)

    # 2. Optimizer state
    if optimizer is not None and state.get("optimizer_state_dict") is not None:
        try:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        except Exception as e:
            print(f"  [WARNING] Optimizer state load skipped: {e}")

    # 3. Scheduler state
    if scheduler is not None and state.get("scheduler_state_dict") is not None:
        try:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        except Exception as e:
            print(f"  [WARNING] Scheduler state load skipped: {e}")

    # 4. Scaler state (AMP)
    if scaler is not None and state.get("scaler_state_dict") is not None:
        try:
            scaler.load_state_dict(state["scaler_state_dict"])
        except Exception as e:
            print(f"  [WARNING] Scaler state load skipped: {e}")

    epoch = state.get("epoch", 0)
    val_metric = state.get("val_metric", state.get("val_dice", 0.0))

    start_epoch = epoch + 1
    print(f"  ✅ Resumed successfully from epoch {epoch} (best metric: {val_metric:.4f})")
    return start_epoch, val_metric


# =============================================================================
# 3. Sliding Window Inference (Guaranteed 100% 3D Volume Coverage)
# =============================================================================

def _get_dim_steps(dim_len: int, patch_len: int, stride_len: int) -> List[int]:
    """Generates coordinate offsets ensuring every voxel along a dimension is covered."""
    if dim_len <= patch_len:
        return [0]
    steps = list(range(0, dim_len - patch_len, stride_len))
    if len(steps) == 0 or steps[-1] != (dim_len - patch_len):
        steps.append(dim_len - patch_len)
    return steps


def _get_gaussian_weight_map(patch_size: Tuple[int, int, int], sigma_scale: float = 0.125) -> torch.Tensor:
    """Generates 3D Gaussian importance weight map to smooth overlapping patch boundaries."""
    ph, pw, pd = patch_size
    sigma_h, sigma_w, sigma_d = ph * sigma_scale, pw * sigma_scale, pd * sigma_scale

    zh = torch.arange(ph, dtype=torch.float32) - (ph - 1) / 2.0
    zw = torch.arange(pw, dtype=torch.float32) - (pw - 1) / 2.0
    zd = torch.arange(pd, dtype=torch.float32) - (pd - 1) / 2.0

    gh = torch.exp(-0.5 * (zh / sigma_h) ** 2)
    gw = torch.exp(-0.5 * (zw / sigma_w) ** 2)
    gd = torch.exp(-0.5 * (zd / sigma_d) ** 2)

    weight = gh.view(-1, 1, 1) * gw.view(1, -1, 1) * gd.view(1, 1, -1)
    weight = weight / weight.max()
    return weight.clamp(min=1e-4)


@torch.no_grad()
def sliding_window_inference(model: nn.Module,
                             volume: torch.Tensor,
                             patch_size: Tuple[int, int, int] = PATCH_SIZE,
                             overlap: Tuple[int, int, int] = PATCH_OVERLAP,
                             num_classes: int = NUM_CLASSES,
                             device: Optional[torch.device] = None,
                             blend_mode: str = "gaussian"
                             ) -> np.ndarray:
    """
    Volumetric 3D sliding-window inference with Gaussian importance blending.
    Guarantees zero truncated slices: covers 100% of spatial dimensions (H, W, D).

    Args:
        model      : Trained PyTorch 3D model
        volume     : (4, H, W, D) or (1, 4, H, W, D) tensor
        patch_size : (ph, pw, pd) 3D patch dimensions
        overlap    : (oh, ow, od) overlap in voxels
        num_classes: 4 (Background + 3 tumor regions)
        blend_mode : 'gaussian' for smooth boundary attenuation or 'uniform'
    Returns:
        pred_seg   : (H, W, D) int64 numpy array of predicted class labels
    """
    model.eval()
    if device is None:
        try:
            device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            device = torch.device("cpu")

    if volume.dim() == 4:
        volume = volume.unsqueeze(0)  # (1, 4, H, W, D)

    _, C, H, W, D = volume.shape
    ph, pw, pd = patch_size
    oh, ow, od = overlap

    sh = max(1, ph - oh)
    sw = max(1, pw - ow)
    sd = max(1, pd - od)

    h_steps = _get_dim_steps(H, ph, sh)
    w_steps = _get_dim_steps(W, pw, sw)
    d_steps = _get_dim_steps(D, pd, sd)

    # Accumulators on CPU to minimize VRAM usage
    pred_prob_sum = torch.zeros((1, num_classes, H, W, D), dtype=torch.float32)
    weight_sum    = torch.zeros((1, 1, H, W, D), dtype=torch.float32)

    if blend_mode == "gaussian":
        patch_weight = _get_gaussian_weight_map(patch_size).unsqueeze(0).unsqueeze(0)  # (1, 1, ph, pw, pd)
    else:
        patch_weight = torch.ones((1, 1, ph, pw, pd), dtype=torch.float32)

    patch_weight_cpu = patch_weight.squeeze(0).squeeze(0)  # (ph, pw, pd)

    for zi in d_steps:
        for yi in h_steps:
            for xi in w_steps:
                ye = yi + ph
                xe = xi + pw
                ze = zi + pd

                # Extract patch
                patch = volume[:, :, yi:ye, xi:xe, zi:ze].to(device)

                logits = model(patch)
                probs = F.softmax(logits, dim=1).cpu()  # (1, C, ph, pw, pd)

                weighted_probs = probs * patch_weight
                pred_prob_sum[:, :, yi:ye, xi:xe, zi:ze] += weighted_probs
                weight_sum[:, :, yi:ye, xi:xe, zi:ze] += patch_weight

    # Normalize by accumulated weights and take argmax
    pred_prob_avg = pred_prob_sum / weight_sum.clamp(min=1e-6)
    pred_seg = pred_prob_avg.argmax(dim=1).squeeze(0).numpy().astype(np.int64)
    return pred_seg


# =============================================================================
# 4. Training & Validation Epoch Loops
# =============================================================================

def train_one_epoch(model: nn.Module,
                    loader,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module,
                    device: torch.device,
                    epoch: int,
                    scaler: Optional[torch.cuda.amp.GradScaler] = None,
                    writer: Optional[SummaryWriter] = None) -> Dict[str, float]:
    """Single full training pass with mixed precision and gradient clipping."""
    model.train()
    total_loss_sum = 0.0
    dice_loss_sum  = 0.0
    sec_loss_sum   = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [Train]", leave=False)
    for images, targets in pbar:
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None and torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss, dl, sl = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss, dl, sl = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss_sum += loss.item()
        dice_loss_sum  += dl.item()
        sec_loss_sum   += sl.item()
        n_batches += 1

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    metrics = {
        "loss":      total_loss_sum / max(n_batches, 1),
        "dice_loss": dice_loss_sum / max(n_batches, 1),
        "sec_loss":  sec_loss_sum / max(n_batches, 1),
    }

    if writer:
        for k, v in metrics.items():
            writer.add_scalar(f"Train/{k}", v, epoch)

    return metrics


@torch.no_grad()
def validate_full_volume(model: nn.Module,
                         loader,
                         criterion: nn.Module,
                         device: torch.device,
                         epoch: int,
                         patch_size: Tuple[int, int, int] = PATCH_SIZE,
                         overlap: Tuple[int, int, int] = PATCH_OVERLAP,
                         writer: Optional[SummaryWriter] = None) -> Dict[str, float]:
    """
    Evaluates validation set using unaugmented full-volume sliding-window inference.
    Computes true volumetric Dice for WT, TC, ET sub-regions.
    """
    from src.metrics import dice_score, get_regions

    model.eval()
    total_loss_sum = 0.0
    wt_dices, tc_dices, et_dices = [], [], []
    n_scans = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [Val]", leave=False)
    for batch in pbar:
        image = batch["image"].squeeze(0)  # (4, H, W, D)
        gt_seg = batch["seg"].squeeze(0).numpy()  # (H, W, D)

        pred_seg = sliding_window_inference(
            model=model,
            volume=image,
            patch_size=patch_size,
            overlap=overlap,
            device=device
        )

        # Compute region masks
        p_reg = get_regions(pred_seg)
        g_reg = get_regions(gt_seg)

        wt_dices.append(dice_score(p_reg["WT"], g_reg["WT"]))
        tc_dices.append(dice_score(p_reg["TC"], g_reg["TC"]))
        et_dices.append(dice_score(p_reg["ET"], g_reg["ET"]))
        n_scans += 1

    wt_mean = float(np.mean(wt_dices)) if wt_dices else 0.0
    tc_mean = float(np.mean(tc_dices)) if tc_dices else 0.0
    et_mean = float(np.mean(et_dices)) if et_dices else 0.0
    mean_dice = (wt_mean + tc_mean + et_mean) / 3.0

    metrics = {
        "wt_dice":   wt_mean,
        "tc_dice":   tc_mean,
        "et_dice":   et_mean,
        "mean_dice": mean_dice,
    }

    if writer:
        for k, v in metrics.items():
            writer.add_scalar(f"Val/{k}", v, epoch)

    return metrics


# =============================================================================
# 5. Helpers
# =============================================================================

def get_scheduler(optimizer: torch.optim.Optimizer,
                  patience: int = 10,
                  factor: float = 0.5) -> ReduceLROnPlateau:
    """Learning rate scheduler monitoring validation WT Dice."""
    return ReduceLROnPlateau(
        optimizer, mode="max", patience=patience, factor=factor,
        min_lr=1e-6, verbose=True
    )


def get_writer(model_name: str, logs_dir: str = LOGS_DIR) -> Optional[SummaryWriter]:
    """TensorBoard SummaryWriter with graceful fallback if tensorboard is not installed."""
    if SummaryWriter is None:
        return None
    log_path = os.path.join(logs_dir, model_name)
    os.makedirs(log_path, exist_ok=True)
    return SummaryWriter(log_path)


def format_time(seconds: float) -> str:
    """Human-readable time formatting."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"
