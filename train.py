#!/usr/bin/env python3
"""
train.py — Unified CLI Training Script for 3D Brain Tumor Segmentation
Supports all 4 model variants (baseline, channel, spatial, hybrid) and robust checkpoint-based resumption.

Usage:
  # Train proposed hybrid attention model:
  python train.py --model_variant hybrid --dataset_path /path/to/BraTS2023

  # Train baseline 3D U-Net:
  python train.py --model_variant baseline --dataset_path /path/to/BraTS2023

  # Resume training from latest checkpoint:
  python train.py --model_variant hybrid --dataset_path /path/to/BraTS2023 --resume

  # Quick test mode (10 patients, 3 epochs):
  python train.py --model_variant hybrid --quick_test
"""

import os
import sys
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from torch.optim import Adam

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATASET_PATH, CHECKPOINT_DIR, LOGS_DIR, RESULTS_DIR,
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    PATCH_SIZE, PATCH_OVERLAP, NUM_WORKERS, PIN_MEMORY,
    LR_PATIENCE, LR_FACTOR, EARLY_STOP_PATIENCE,
    RANDOM_SEED, get_device, ensure_directories
)
from src.dataset import get_dataloaders
from src.models import build_model, count_parameters
from src.losses import CombinedLoss, compute_class_weights_from_train_split
from src.utils import (
    set_seed, save_checkpoint, load_checkpoint,
    train_one_epoch, validate_full_volume,
    get_scheduler, get_writer, format_time
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified 3D Brain Tumor Segmentation Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model & Architecture
    parser.add_argument("--model_variant", type=str, default="hybrid",
                        choices=["baseline", "channel", "spatial", "hybrid"],
                        help="Model architecture variant for training / ablation study")
    parser.add_argument("--init_features", type=int, default=32,
                        help="Number of feature channels in first encoder stage")

    # Paths & Directory structure
    parser.add_argument("--dataset_path", type=str, default=DATASET_PATH,
                        help="Path to BraTS dataset directory containing patient folders")
    parser.add_argument("--checkpoint_dir", type=str, default=CHECKPOINT_DIR,
                        help="Directory to save/load model checkpoints")
    parser.add_argument("--logs_dir", type=str, default=LOGS_DIR,
                        help="Directory for TensorBoard telemetry logs")
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR,
                        help="Directory for output metrics and evaluation tables")

    # Optimization Hyperparameters
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help="Total training epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Mini-batch size for training patch extraction")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help="Initial learning rate for Adam optimizer")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY,
                        help="L2 regularization weight decay")
    parser.add_argument("--patch_size", type=int, nargs=3, default=list(PATCH_SIZE),
                        help="3D training patch dimensions (H W D)")

    # Data Partitioning
    parser.add_argument("--fold", type=int, default=0,
                        help="Cross-validation fold index (0 to n_folds - 1)")
    parser.add_argument("--n_folds", type=int, default=3,
                        help="Total number of K-Fold partitions")
    parser.add_argument("--max_patients", type=int, default=None,
                        help="Cap on maximum patient scans to load")

    # Hardware & Performance
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS,
                        help="Number of DataLoader CPU worker processes")
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Enable Automatic Mixed Precision (AMP) for accelerated training")
    parser.add_argument("--no_amp", action="store_false", dest="amp",
                        help="Disable Automatic Mixed Precision")

    # Execution Modes & Resumption
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume training from latest checkpoint")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Explicit checkpoint file path to resume from")
    parser.add_argument("--quick_test", action="store_true", default=False,
                        help="Quick test mode: runs on 10 patients for 3 epochs")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help="Master random seed for reproducibility")

    return parser.parse_args()


def main():
    args = parse_args()
    patch_size = tuple(args.patch_size)

    # 1. Initialize environment & seeds
    set_seed(args.seed)
    ensure_directories([args.checkpoint_dir, args.logs_dir, args.results_dir])
    device = get_device()

    print("\n" + "=" * 70)
    print("  🧠 3D Brain Tumor Segmentation — Training Engine")
    print(f"  Model Variant  : {args.model_variant.upper()}")
    print(f"  Device         : {device}")
    print(f"  Patch Size     : {patch_size}")
    print(f"  Mixed Precision: {args.amp}")
    print(f"  Checkpoint Dir : {args.checkpoint_dir}")
    print("=" * 70 + "\n")

    # 2. Build DataLoaders (Patient-level zero-leakage guaranteed)
    print("📦 Initializing Subject-Grouped DataLoaders...")
    train_loader, val_loader, test_loader, test_folders = get_dataloaders(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=PIN_MEMORY,
        fold=args.fold,
        n_folds=args.n_folds,
        max_patients=args.max_patients,
        quick_test=args.quick_test,
        quick_test_n=10,
        seed=args.seed
    )

    # 3. Instantiate Neural Network Model
    model_name = f"{args.model_variant}_unet3d_fold{args.fold}"
    model = build_model(
        variant=args.model_variant,
        in_channels=4,
        out_channels=4,
        init_features=args.init_features
    ).to(device)

    param_info = count_parameters(model)
    print(f"  Model Name  : {model_name}")
    print(f"  Parameters  : {param_info['formatted']}\n")

    # 4. Optimizer, Loss & Scheduler
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = CombinedLoss(num_classes=4, dice_weight=0.5, ce_weight=0.5, use_focal=True)
    scheduler = get_scheduler(optimizer, patience=LR_PATIENCE, factor=LR_FACTOR)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and torch.cuda.is_available()))
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and torch.cuda.is_available()))
    writer = get_writer(model_name=model_name, logs_dir=args.logs_dir)

    # 5. Checkpoint Resumption
    start_epoch = 0
    best_val_dice = 0.0
    if args.resume or args.checkpoint_path:
        start_epoch, best_val_dice = load_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            checkpoint_path=args.checkpoint_path,
            model_name=model_name,
            checkpoint_dir=args.checkpoint_dir,
            prefer_best=False,
            device=device
        )

    max_epochs = 3 if args.quick_test else args.epochs
    if start_epoch >= max_epochs:
        print(f"✅ Training already complete ({start_epoch}/{max_epochs} epochs finished).")
        return

    # 6. Training Loop
    print(f"🚀 Launching Training: Epoch {start_epoch + 1} → {max_epochs}")
    no_improve_count = 0
    total_start_time = time.time()

    for epoch in range(start_epoch, max_epochs):
        epoch_start_time = time.time()

        # ── Train ────────────────────────────────────────────────────────────
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch + 1,
            scaler=scaler if args.amp else None,
            writer=writer
        )

        # ── Validate (Full Volume Sliding Window) ────────────────────────────
        val_metrics = validate_full_volume(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch + 1,
            patch_size=patch_size,
            overlap=PATCH_OVERLAP,
            writer=writer
        )

        val_wt_dice = val_metrics["wt_dice"]
        val_mean_dice = val_metrics["mean_dice"]

        # Step Learning Rate Scheduler
        scheduler.step(val_wt_dice)
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("Train/LearningRate", current_lr, epoch + 1)

        # ── Checkpoint Saving ────────────────────────────────────────────────
        is_best = val_wt_dice > best_val_dice
        if is_best:
            best_val_dice = val_wt_dice
            no_improve_count = 0
        else:
            no_improve_count += 1

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch + 1,
            val_metric=val_wt_dice,
            model_name=model_name,
            checkpoint_dir=args.checkpoint_dir,
            is_best=is_best,
            extra_info={"val_mean_dice": val_mean_dice, "fold": args.fold, "variant": args.model_variant}
        )

        epoch_elapsed = time.time() - epoch_start_time
        remaining_epochs = max_epochs - (epoch + 1)
        eta_seconds = epoch_elapsed * remaining_epochs

        star = " ★ NEW BEST" if is_best else ""
        print(f"Epoch [{epoch+1:03d}/{max_epochs:03d}] "
              f"| Train Loss: {train_metrics['loss']:.4f} "
              f"| Val WT Dice: {val_wt_dice:.4f} "
              f"| Val Mean: {val_mean_dice:.4f} "
              f"| Best WT: {best_val_dice:.4f} "
              f"| LR: {current_lr:.1e} "
              f"| Time: {epoch_elapsed:.1f}s (ETA: {format_time(eta_seconds)}){star}")

        # ── Early Stopping Check ─────────────────────────────────────────────
        if no_improve_count >= EARLY_STOP_PATIENCE:
            print(f"\n⚠️  Early stopping triggered at epoch {epoch+1} "
                  f"(no validation improvement for {EARLY_STOP_PATIENCE} epochs).")
            break

    writer.close()
    total_time = time.time() - total_start_time
    print("\n" + "=" * 70)
    print(f"  🏁 Training Completed for {model_name}")
    print(f"  Best Validation WT Dice : {best_val_dice:.4f}")
    print(f"  Total Elapsed Time      : {format_time(total_time)}")
    print(f"  Checkpoints Saved To    : {args.checkpoint_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
