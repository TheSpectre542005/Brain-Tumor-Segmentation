"""
evaluate_models.py — Standalone Evaluation & Head-to-Head Comparison Script

Evaluates trained models (Baseline 3D U-Net vs Proposed 3D Attention U-Net)
on the held-out test set with zero subject leakage.

Computes:
  - Dice Score (WT, TC, ET)
  - IoU / Jaccard (WT, TC, ET)
  - 95% Hausdorff Distance HD95 (WT, TC, ET)
  - Sensitivity & Specificity

Outputs:
  - results/comparison_table.csv
  - results/comparison_summary.txt
  - results/dice_comparison_barchart.png
  - results/dice_boxplots.png
"""

import sys, os, time
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Headless mode for server / DGX
import matplotlib.pyplot as plt
import torch
import nibabel as nib
from tqdm import tqdm

from src.models import UNet3D, AttentionUNet3D
from src.dataset import get_dataloaders, load_patient, get_file_paths
from src.metrics import compute_patient_metrics, aggregate_metrics, print_metrics_table
from src.utils import load_checkpoint, sliding_window_inference, set_seed
from src.config import (
    DATASET_PATH, CHECKPOINT_DIR, RESULTS_DIR, DEVICE,
    PATCH_SIZE, PATCH_OVERLAP, N_FOLDS, MAX_PATIENTS, RANDOM_SEED
)

set_seed(RANDOM_SEED)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Define Candidate Models
# ---------------------------------------------------------------------------
CANDIDATE_CONFIGS = [
    {
        "name": "Baseline 3D U-Net",
        "short_name": "Baseline",
        "model_name": "baseline_unet3d",
        "model_fn": lambda: UNet3D(in_channels=4, out_channels=4, init_features=32),
    },
    {
        "name": "Attention U-Net (Full — Proposed)",
        "short_name": "Attn U-Net (Ours)",
        "model_name": "attention_unet3d_full",
        "model_fn": lambda: AttentionUNet3D(in_channels=4, out_channels=4, init_features=32,
                                           use_channel_attn=True, use_spatial_attn=True),
    },
    {
        "name": "Attention U-Net (Channel only)",
        "short_name": "Ch. Attn",
        "model_name": "attention_unet3d_channel_only",
        "model_fn": lambda: AttentionUNet3D(in_channels=4, out_channels=4, init_features=32,
                                           use_channel_attn=True, use_spatial_attn=False),
    },
    {
        "name": "Attention U-Net (Spatial only)",
        "short_name": "Sp. Attn",
        "model_name": "attention_unet3d_spatial_only",
        "model_fn": lambda: AttentionUNet3D(in_channels=4, out_channels=4, init_features=32,
                                           use_channel_attn=False, use_spatial_attn=True),
    },
]

# Auto-detect which models actually have checkpoints saved on disk
ACTIVE_MODELS = []
print(f"Device: {DEVICE}")
print(f"Searching checkpoints in: {CHECKPOINT_DIR}")

for cfg in CANDIDATE_CONFIGS:
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{cfg['model_name']}_best.pth")
    if os.path.exists(ckpt_path):
        model = cfg["model_fn"]().to(DEVICE)
        epoch, best_dice = load_checkpoint(model, optimizer=None, model_name=cfg["model_name"], prefer_best=True)
        model.eval()
        ACTIVE_MODELS.append({
            "name": cfg["name"],
            "short_name": cfg["short_name"],
            "model_name": cfg["model_name"],
            "model": model,
            "best_dice": best_dice,
            "epoch": epoch
        })
        print(f"  [FOUND] {cfg['name']} (Best Val Dice: {best_dice:.4f}, Epoch {epoch})")
    else:
        print(f"  [SKIP]  {cfg['name']} (No checkpoint at {ckpt_path})")

if not ACTIVE_MODELS:
    raise RuntimeError(f"No trained model checkpoints found in {CHECKPOINT_DIR}!")

# ---------------------------------------------------------------------------
# 2. Get Test Patient Folders (Subject-Grouped, Zero Leakage)
# ---------------------------------------------------------------------------
print("\nLoading dataset and creating Subject-Grouped test split...")
_, _, _, test_folders = get_dataloaders(
    dataset_path=DATASET_PATH,
    batch_size=1,
    num_workers=2,
    fold=0,
    n_folds=N_FOLDS,
    max_patients=MAX_PATIENTS,
    quick_test=False,
)
print(f"Held-out Test Set: {len(test_folders)} patients (never seen during training/val)")

# ---------------------------------------------------------------------------
# 3. Run Inference and Compute Metrics
# ---------------------------------------------------------------------------
all_model_results = {}

for cfg in ACTIVE_MODELS:
    print(f"\n{'='*70}")
    print(f"Evaluating: {cfg['name']}")
    print(f"{'='*70}")

    patient_results = []
    t_start = time.time()

    for folder in tqdm(test_folders, desc=cfg["short_name"]):
        image, gt_seg = load_patient(folder, has_seg=True)
        image_tensor = torch.from_numpy(image.astype(np.float32))

        paths = get_file_paths(folder)
        nii = nib.load(paths["flair"])
        voxel_spacing = tuple(abs(float(z)) for z in nii.header.get_zooms()[:3])

        with torch.no_grad():
            pred_seg = sliding_window_inference(
                cfg["model"], image_tensor,
                patch_size=PATCH_SIZE,
                overlap=PATCH_OVERLAP,
                device=DEVICE
            )

        metrics = compute_patient_metrics(pred_seg, gt_seg, voxel_spacing)
        patient_results.append(metrics)

    all_model_results[cfg["name"]] = patient_results
    elapsed = time.time() - t_start
    print(f"Completed in {elapsed/60:.1f} minutes")

    agg = aggregate_metrics(patient_results)
    print_metrics_table(agg, model_name=cfg["name"])

# ---------------------------------------------------------------------------
# 4. Generate Comparison Table & CSV
# ---------------------------------------------------------------------------
rows = []
for cfg in ACTIVE_MODELS:
    agg = aggregate_metrics(all_model_results[cfg["name"]])
    row = {"Model": cfg["name"]}
    for region in ["WT", "TC", "ET"]:
        for metric in ["dice", "iou", "hd95", "sensitivity", "specificity"]:
            key = f"{metric.upper()} {region}"
            row[key] = round(agg[region][metric]["mean"], 4)
    rows.append(row)

results_df = pd.DataFrame(rows)
csv_path = os.path.join(RESULTS_DIR, "comparison_table.csv")
results_df.to_csv(csv_path, index=False)
print(f"\nSaved CSV to: {csv_path}")

summary_cols = ["Model", "DICE WT", "DICE TC", "DICE ET", "HD95 WT", "HD95 TC", "HD95 ET"]
print("\n" + "="*80)
print("FINAL HEAD-TO-HEAD COMPARISON TABLE")
print("="*80)
print(results_df[[c for c in summary_cols if c in results_df.columns]].to_string(index=False))

# ---------------------------------------------------------------------------
# 5. Generate Comparison Bar Chart
# ---------------------------------------------------------------------------
model_names = [cfg["name"] for cfg in ACTIVE_MODELS]
short_names = [cfg["short_name"] for cfg in ACTIVE_MODELS]
regions = ["WT", "TC", "ET"]
region_colors = ["#2b5c8f", "#d9534f", "#2ca02c"]

x = np.arange(len(model_names))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
for i, (region, color) in enumerate(zip(regions, region_colors)):
    dice_means = results_df[f"DICE {region}"].values
    bars = ax.bar(x + i * width, dice_means, width, label=f"Dice {region}",
                  color=color, alpha=0.88, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, dice_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xticks(x + width)
ax.set_xticklabels(short_names, fontsize=12, fontweight="bold")
ax.set_ylabel("Dice Score (DSC)", fontsize=12)
ax.set_ylim(0.70, 1.0)
ax.set_title("Test Set Dice Score Comparison: Baseline vs Proposed Attention U-Net",
             fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=11, loc="lower right")
ax.grid(True, alpha=0.3, axis="y")
ax.spines[["top", "right"]].set_visible(False)

bar_chart_path = os.path.join(RESULTS_DIR, "dice_comparison_barchart.png")
plt.tight_layout()
plt.savefig(bar_chart_path, bbox_inches="tight")
plt.close()
print(f"Saved Bar Chart to: {bar_chart_path}")

# ---------------------------------------------------------------------------
# 6. Generate Box Plots
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=200)
fig.suptitle("Dice Score Distribution Across Test Patients", fontsize=14, fontweight="bold")

for ax, region, color in zip(axes, regions, region_colors):
    data_per_model = []
    for cfg in ACTIVE_MODELS:
        patient_dices = [r[region]["dice"] for r in all_model_results[cfg["name"]]]
        data_per_model.append(patient_dices)

    bp = ax.boxplot(data_per_model, patch_artist=True, notch=False,
                    medianprops={"color": "black", "linewidth": 2})

    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(range(1, len(ACTIVE_MODELS) + 1))
    ax.set_xticklabels(short_names, fontsize=10, fontweight="bold")
    ax.set_ylabel("Dice Score")
    ax.set_title(f"{region} Sub-region", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.spines[["top", "right"]].set_visible(False)

box_plot_path = os.path.join(RESULTS_DIR, "dice_boxplots.png")
plt.tight_layout()
plt.savefig(box_plot_path, bbox_inches="tight")
plt.close()
print(f"Saved Box Plot to: {box_plot_path}")

print("\nEvaluation successfully completed! All results and plots saved to results/")
