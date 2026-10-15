#!/usr/bin/env python3
"""
train.py — Command-Line Training Script for Brain Tumor Segmentation (BraTS 2023 GLI)
Optimized for NVIDIA Blackwell B200 / DGX Cluster.

Usage:
    # 1. Quick Test (10 patients, 3 epochs — verifies pipeline in ~3 min):
    python3 train.py --model baseline --quick-test

    # 2. Full Training Baseline 3D U-Net:
    python3 train.py --model baseline

    # 3. Full Training Proposed Dual-Attention 3D U-Net:
    python3 train.py --model attention

    # 4. Background Training with log (immune to terminal close):
    nohup python3 train.py --model baseline > train_baseline.log 2>&1 &
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.optim import Adam

from src.config import (
    DATASET_PATH, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, WEIGHT_DECAY,
    MAX_PATIENTS, N_FOLDS, FOLD, DEVICE, RANDOM_SEED,
    EARLY_STOP_PATIENCE, LR_PATIENCE, LR_FACTOR, USE_AMP, NUM_WORKERS
)
from src.models import UNet3D, AttentionUNet3D
from src.losses import CombinedLoss
from src.dataset import get_dataloaders
from src.utils import (
    set_seed, save_checkpoint, load_checkpoint,
    train_one_epoch, validate_one_epoch,
    get_scheduler, get_writer, format_time
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train 3D Brain Tumor Segmentation Models on B200")
    parser.add_argument(
        "--model", type=str, choices=["baseline", "attention"], default="attention",
        help="Which model to train: 'baseline' (UNet3D) or 'attention' (Dual-Attention UNet3D, default)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of epochs to train (overrides config.py if specified)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--workers", type=int, default=NUM_WORKERS,
        help=f"DataLoader workers (default: {NUM_WORKERS})"
    )
    parser.add_argument(
        "--quick-test", action="store_true",
        help="Run quick test mode (10 patients, 3 epochs) to verify pipeline"
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Do NOT resume from checkpoint; train from scratch"
    )
    parser.add_argument(
        "--fold", type=int, default=FOLD,
        help=f"K-Fold split index (default: {FOLD})"
    )
    parser.add_argument(
        "--dataset-path", type=str, default=DATASET_PATH,
        help="Custom dataset path if different from config.py"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(RANDOM_SEED)

    print("=" * 70)
    print("  BraTS 2023 GLI Brain Tumor Segmentation Training")
    print(f"  Target Device : {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"  Model Choice  : {args.model.upper()}")
    print(f"  Quick Test    : {'YES (10 patients, 3 epochs)' if args.quick_test else 'NO (Full Training)'}")
    print(f"  AMP Enabled   : {USE_AMP} (BF16)")
    print("=" * 70)

    # 1. Load Data
    print("\n[1/4] Loading DataLoader...")
    train_loader, val_loader, test_loader, _ = get_dataloaders(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        num_workers=args.workers,
        fold=args.fold,
        n_folds=N_FOLDS,
        max_patients=MAX_PATIENTS,
        quick_test=args.quick_test,
        quick_test_n=10,
    )
    print(f"  Train Batches: {len(train_loader)} | Val Batches: {len(val_loader)}")

    # 2. Build Model
    print("\n[2/4] Initializing Model...")
    if args.model == "baseline":
        model_name = "baseline_unet3d"
        model = UNet3D(in_channels=4, out_channels=4, init_features=32).to(DEVICE)
    else:
        model_name = "attention_unet3d"
        model = AttentionUNet3D(in_channels=4, out_channels=4, init_features=32).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Architecture: {model_name} ({total_params:.1f}M parameters)")

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = CombinedLoss(num_classes=4, dice_weight=0.5, focal_weight=0.5)
    scheduler = get_scheduler(optimizer, patience=LR_PATIENCE, factor=LR_FACTOR)
    writer = get_writer(model_name)

    # 3. Checkpoint Resumption
    start_epoch = 0
    best_val_dice = 0.0
    if not args.no_resume:
        print("\n[3/4] Checking for checkpoints to resume...")
        start_epoch, best_val_dice = load_checkpoint(
            model, optimizer, model_name=model_name, prefer_best=False
        )
        if start_epoch > 0:
            print(f"  Resumed from epoch {start_epoch} (Best Val Dice: {best_val_dice:.4f})")
        else:
            print("  No previous checkpoint found. Starting fresh.")
    else:
        print("\n[3/4] Starting fresh (resume disabled).")

    # Determine epoch limit
    if args.quick_test:
        max_epochs = 3
    elif args.epochs is not None:
        max_epochs = args.epochs
    else:
        max_epochs = NUM_EPOCHS

    # 4. Training Loop
    print(f"\n[4/4] Starting Training: Epoch {start_epoch + 1} -> {max_epochs}")
    training_start = time.time()
    no_improve_count = 0

    try:
        for epoch in range(start_epoch, max_epochs):
            epoch_start = time.time()

            # Train pass
            train_metrics = train_one_epoch(
                model, train_loader, optimizer, criterion, DEVICE, epoch, writer
            )

            # Validation pass
            val_metrics = validate_one_epoch(
                model, val_loader, criterion, DEVICE, epoch, writer
            )

            # Scheduler step (monitors Whole Tumor Dice)
            scheduler.step(val_metrics["wt_dice"])

            # Checkpoint save
            is_best = val_metrics["wt_dice"] > best_val_dice
            if is_best:
                best_val_dice = val_metrics["wt_dice"]
                no_improve_count = 0
            else:
                no_improve_count += 1

            save_checkpoint(
                model, optimizer, epoch, val_metrics["wt_dice"],
                model_name=model_name, is_best=is_best
            )

            epoch_time = time.time() - epoch_start
            eta = epoch_time * (max_epochs - epoch - 1)

            print(
                f"Epoch [{epoch+1:3d}/{max_epochs:3d}] "
                f"| Train Loss: {train_metrics['loss']:.4f} "
                f"| Val Loss: {val_metrics['loss']:.4f} "
                f"| Val WT Dice: {val_metrics['wt_dice']:.4f} "
                f"| Best: {best_val_dice:.4f} "
                f"| Time: {format_time(epoch_time)} "
                f"| ETA: {format_time(eta)}"
            )

            if no_improve_count >= EARLY_STOP_PATIENCE:
                print(f"\n⚠️  Early stopping triggered at epoch {epoch+1} (no gain in {EARLY_STOP_PATIENCE} epochs).")
                break

    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user! Checkpoint safely saved.")
    finally:
        writer.close()

    total_time = time.time() - training_start
    print("=" * 70)
    print(f"🏁 Training session finished!")
    print(f"   Best Val WT Dice: {best_val_dice:.4f}")
    print(f"   Total Time      : {format_time(total_time)}")
    print(f"   Checkpoints in  : ./checkpoints/{model_name}_best.pth")
    print("=" * 70)


if __name__ == "__main__":
    main()
