"""
run_ablation.py — Standalone Ablation Study Pipeline

Quantifies the incremental contribution of each attention component:
1. Baseline 3D U-Net             (No Attention)
2. Channel Attention Only (SE)   (Squeeze-and-Excitation blocks)
3. Spatial Attention Only (Gates)(Spatial Attention Gates)
4. Dual-Attention U-Net (Full)   (Proposed SE + Spatial Gates)

Features:
- Auto-detects existing checkpoints in checkpoints/
- Trains missing ablation variants if --train-missing is passed
- Computes comprehensive test metrics (Dice, HD95, Sensitivity) across WT, TC, ET
- Conducts paired Wilcoxon signed-rank significance tests (p < 0.05)
- Generates publication-ready CSV and comparison bar chart

Usage:
  # Check status & evaluate available models:
  python3 run_ablation.py

  # Train any missing ablation models in background, then evaluate:
  nohup python3 run_ablation.py --train-missing > ablation.log 2>&1 &
"""

import sys, os, time, argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend for DGX / server
import matplotlib.pyplot as plt
from scipy import stats
import torch
from torch.optim import Adam

from src.models import UNet3D, AttentionUNet3D
from src.dataset import get_dataloaders, load_patient, get_file_paths
from src.losses import CombinedLoss
from src.metrics import compute_patient_metrics, aggregate_metrics
from src.utils import (
    load_checkpoint, save_checkpoint, sliding_window_inference,
    train_one_epoch, validate_one_epoch, get_scheduler, set_seed
)
from src.config import (
    DATASET_PATH, CHECKPOINT_DIR, RESULTS_DIR, DEVICE,
    PATCH_SIZE, PATCH_OVERLAP, RANDOM_SEED, MAX_PATIENTS,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, LR_PATIENCE, LR_FACTOR,
    EARLY_STOP_PATIENCE, USE_AMP, NUM_WORKERS
)

# ---------------------------------------------------------------------------
# Setup & Definitions
# ---------------------------------------------------------------------------
set_seed(RANDOM_SEED)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

ABLATION_CONFIGS = [
    {
        "name": "Baseline (No Attention)",
        "short_name": "Baseline",
        "model_name": "baseline_unet3d",
        "ch_attn": False,
        "sp_attn": False,
        "model_fn": lambda: UNet3D(in_channels=4, out_channels=4, init_features=32),
    },
    {
        "name": "Channel Attention Only (SE)",
        "short_name": "+ Channel (SE)",
        "model_name": "attention_unet3d_channel_only",
        "ch_attn": True,
        "sp_attn": False,
        "model_fn": lambda: AttentionUNet3D(in_channels=4, out_channels=4, init_features=32,
                                           use_channel_attn=True, use_spatial_attn=False),
    },
    {
        "name": "Spatial Attention Only (Gates)",
        "short_name": "+ Spatial (Gates)",
        "model_name": "attention_unet3d_spatial_only",
        "ch_attn": False,
        "sp_attn": True,
        "model_fn": lambda: AttentionUNet3D(in_channels=4, out_channels=4, init_features=32,
                                           use_channel_attn=False, use_spatial_attn=True),
    },
    {
        "name": "Dual-Attention U-Net (Full — Proposed)",
        "short_name": "Full (Ours)",
        "model_name": "attention_unet3d_full",
        "ch_attn": True,
        "sp_attn": True,
        "model_fn": lambda: AttentionUNet3D(in_channels=4, out_channels=4, init_features=32,
                                           use_channel_attn=True, use_spatial_attn=True),
    },
]

def check_checkpoints():
    """Verify which model checkpoints exist."""
    status = {}
    for cfg in ABLATION_CONFIGS:
        best_path = os.path.join(CHECKPOINT_DIR, f"{cfg['model_name']}_best.pth")
        status[cfg["model_name"]] = os.path.exists(best_path)
    return status

def train_variant(cfg, train_loader, val_loader, epochs=40):
    """Train a missing ablation variant."""
    m_name = cfg["model_name"]
    print("\n" + "=" * 70)
    print(f"  Training Ablation Variant: {cfg['name']}")
    print(f"  Model name: {m_name} | Target epochs: {epochs}")
    print("=" * 70)

    model = cfg["model_fn"]().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = CombinedLoss(num_classes=4, dice_weight=0.5, focal_weight=0.5)
    scheduler = get_scheduler(optimizer, patience=LR_PATIENCE, factor=LR_FACTOR)

    start_epoch, best_val_dice = load_checkpoint(model, optimizer, model_name=m_name, prefer_best=False)
    no_improve_count = 0

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        train_m = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, epoch)
        val_m   = validate_one_epoch(model, val_loader, criterion, DEVICE, epoch)
        ep_time = time.time() - t0

        val_dice = val_m["wt_dice"]
        is_best = val_dice > best_val_dice
        if is_best:
            best_val_dice = val_dice
            no_improve_count = 0
        else:
            no_improve_count += 1

        save_checkpoint(epoch, model, optimizer, val_dice, m_name, is_best=is_best)
        scheduler.step(val_dice)

        print(f"  Epoch [{epoch+1:2d}/{epochs:2d}] ({ep_time:.1f}s) | "
              f"Train Loss: {train_m['total']:.4f} | Val WT Dice: {val_dice:.4f} | "
              f"Best: {best_val_dice:.4f} {'🔥 BEST' if is_best else ''}")

        if no_improve_count >= EARLY_STOP_PATIENCE:
            print(f"  🛑 Early stopping triggered at epoch {epoch+1}")
            break

    print(f"  ✅ Completed training for {cfg['name']}. Best Val Dice: {best_val_dice:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Run BraTS 2023 Ablation Study")
    parser.add_argument("--train-missing", action="store_true",
                        help="Automatically train any missing ablation variants before evaluation")
    parser.add_argument("--epochs", type=int, default=40,
                        help="Epochs for training missing variants (default: 40)")
    parser.add_argument("--quick-eval", type=int, default=None,
                        help="Limit test evaluation to N patients for rapid test")
    args = parser.parse_args()

    print("=" * 70)
    print("  BraTS 2023 GLI — Ablation Study Pipeline")
    print(f"  Device: {DEVICE}")
    print("=" * 70)

    # 1. Check existing checkpoints
    ckpt_status = check_checkpoints()
    print("\n[1/4] Checkpoint Status:")
    for cfg in ABLATION_CONFIGS:
        present = ckpt_status[cfg["model_name"]]
        print(f"  {'✅' if present else '❌'} {cfg['name']:42s} : "
              f"{'Found' if present else 'MISSING (' + cfg['model_name'] + '_best.pth)'}")

    # 2. Train missing if requested
    missing_models = [cfg for cfg in ABLATION_CONFIGS if not ckpt_status[cfg["model_name"]]]
    if missing_models:
        if args.train_missing:
            print(f"\n[2/4] Training {len(missing_models)} missing variants...")
            train_loader, val_loader, _, _ = get_dataloaders(
                dataset_path=DATASET_PATH,
                batch_size=BATCH_SIZE,
                num_workers=NUM_WORKERS,
                max_patients=MAX_PATIENTS,
            )
            for m_cfg in missing_models:
                train_variant(m_cfg, train_loader, val_loader, epochs=args.epochs)
        else:
            print(f"\n⚠️  {len(missing_models)} ablation model(s) are missing.")
            print("   Pass '--train-missing' to train them automatically in the background:")
            print("   nohup python3 run_ablation.py --train-missing > ablation.log 2>&1 &")
            print("\n   Proceeding to evaluate whatever models are currently available...")

    # 3. Evaluate available ablation models
    print("\n[3/4] Evaluating available ablation models on held-out test scans...")
    _, _, test_loader, test_folders = get_dataloaders(
        dataset_path=DATASET_PATH,
        batch_size=1,
        num_workers=2,
        n_folds=3,
        fold=0,
        max_patients=MAX_PATIENTS,
    )
    if args.quick_eval:
        test_folders = test_folders[:args.quick_eval]
    print(f"  Total test patients for evaluation: {len(test_folders)}")

    all_ablation_results = {}
    evaluated_cfgs = []

    for cfg in ABLATION_CONFIGS:
        m_name = cfg["model_name"]
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"{m_name}_best.pth")
        if not os.path.exists(ckpt_path):
            print(f"  Skipping {cfg['name']} (checkpoint not found)")
            continue

        print(f"\n  Evaluating: {cfg['name']}...")
        model = cfg["model_fn"]().to(DEVICE)
        load_checkpoint(model, model_name=m_name, prefer_best=True)
        model.eval()

        patient_metrics = []
        t0 = time.time()
        for folder in test_folders:
            image, gt_seg = load_patient(folder, has_seg=True)
            img_tensor = torch.from_numpy(image.astype(np.float32))
            with torch.no_grad():
                pred = sliding_window_inference(model, img_tensor, PATCH_SIZE, PATCH_OVERLAP, DEVICE)
            m = compute_patient_metrics(pred, gt_seg)
            patient_metrics.append(m)

        eval_time = time.time() - t0
        all_ablation_results[cfg["name"]] = patient_metrics
        evaluated_cfgs.append(cfg)

        agg = aggregate_metrics(patient_metrics)
        print(f"     Time: {eval_time/60:.1f}m | "
              f"WT Dice: {agg['WT']['dice']['mean']:.4f} | "
              f"TC Dice: {agg['TC']['dice']['mean']:.4f} | "
              f"ET Dice: {agg['ET']['dice']['mean']:.4f}")

    # 4. Generate Report & Plots
    print("\n[4/4] Generating Ablation Table & Comparison Plots...")
    rows = []
    for cfg in evaluated_cfgs:
        name = cfg["name"]
        agg = aggregate_metrics(all_ablation_results[name])
        model_obj = cfg["model_fn"]()
        param_count = sum(p.numel() for p in model_obj.parameters()) / 1e6

        rows.append({
            "Model Variant": name,
            "Short Name": cfg["short_name"],
            "Params (M)": round(param_count, 2),
            "Channel Attn (SE)": "Yes" if cfg["ch_attn"] else "No",
            "Spatial Attn (Gates)": "Yes" if cfg["sp_attn"] else "No",
            "Dice WT": round(agg["WT"]["dice"]["mean"], 4),
            "Dice TC": round(agg["TC"]["dice"]["mean"], 4),
            "Dice ET": round(agg["ET"]["dice"]["mean"], 4),
            "Sens WT": round(agg["WT"]["sensitivity"]["mean"], 4),
            "Sens TC": round(agg["TC"]["sensitivity"]["mean"], 4),
            "Sens ET": round(agg["ET"]["sensitivity"]["mean"], 4),
            "HD95 WT": round(agg["WT"]["hd95"]["mean"], 2),
            "HD95 ET": round(agg["ET"]["hd95"]["mean"], 2),
        })

    ablation_df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "ablation_study.csv")
    ablation_df.to_csv(csv_path, index=False)
    print(f"\n  [Saved] {csv_path}")

    # Print summary table
    print("\n" + "=" * 90)
    print("  Ablation Study Summary Table")
    print("=" * 90)
    print(ablation_df[["Model Variant", "Channel Attn (SE)", "Spatial Attn (Gates)", "Dice WT", "Dice TC", "Dice ET", "Sens ET"]].to_string(index=False))

    # Statistical significance test if baseline and full are present
    bl_name = "Baseline (No Attention)"
    fl_name = "Dual-Attention U-Net (Full — Proposed)"
    if bl_name in all_ablation_results and fl_name in all_ablation_results:
        print("\n" + "-" * 70)
        print("  Statistical Significance (Paired Wilcoxon Signed-Rank Test):")
        print("  Null hypothesis: Proposed <= Baseline | Alternative: Proposed > Baseline")
        print("-" * 70)
        bl_res = all_ablation_results[bl_name]
        fl_res = all_ablation_results[fl_name]
        for region in ["WT", "TC", "ET"]:
            bl_dices = [r[region]["dice"] for r in bl_res]
            fl_dices = [r[region]["dice"] for r in fl_res]
            try:
                stat, p_val = stats.wilcoxon(fl_dices, bl_dices, alternative="greater")
                sig = "✅ Statistically Significant (p < 0.05)" if p_val < 0.05 else "Not statistically significant"
                print(f"  {region} Dice: Baseline={np.mean(bl_dices):.4f} -> Proposed={np.mean(fl_dices):.4f} | p={p_val:.4e} | {sig}")
            except Exception as e:
                print(f"  {region} Wilcoxon test: {e}")

    # Generate Ablation Bar Chart
    if len(evaluated_cfgs) >= 2:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Ablation Study — Impact of Attention Modules on Segmentation Performance",
                     fontsize=14, fontweight="bold", y=1.02)

        labels = [c["short_name"] for c in evaluated_cfgs]
        colors = ["#7f8c8d", "#2980b9", "#8e44ad", "#27ae60"][:len(evaluated_cfgs)]

        for ax, region in zip(axes, ["WT", "TC", "ET"]):
            dices = [ablation_df.loc[ablation_df["Model Variant"] == c["name"], f"Dice {region}"].values[0] for c in evaluated_cfgs]
            bars = ax.bar(labels, dices, color=colors, edgecolor="black", linewidth=0.8, width=0.55)
            ax.set_title(f"{region} Dice Score", fontsize=12, fontweight="bold", pad=8)
            ax.set_ylim(0.0, 1.05)
            ax.grid(axis="y", linestyle="--", alpha=0.5)

            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.3f}",
                        ha="center", va="bottom", fontsize=10, fontweight="bold")

        plt.tight_layout()
        chart_path = os.path.join(RESULTS_DIR, "ablation_barchart.png")
        plt.savefig(chart_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Saved] {chart_path}")

    print("\n" + "=" * 70)
    print("  Ablation Pipeline Finished Successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
