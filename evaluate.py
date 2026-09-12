#!/usr/bin/env python3
"""
evaluate.py — Standalone Evaluation & Comparative Metrics Suite
Runs unaugmented full-volume sliding-window inference on held-out test cases,
computes the complete 15-metric suite (WT, TC, ET), and outputs report-ready tables.

Usage:
  # Evaluate a trained model on the test split:
  python evaluate.py --checkpoint checkpoints/hybrid_unet3d_fold0_best.pth --model_variant hybrid

  # Compare proposed model against baseline with Wilcoxon paired test:
  python evaluate.py --checkpoint checkpoints/hybrid_unet3d_fold0_best.pth --model_variant hybrid \
                     --compare_baseline checkpoints/baseline_unet3d_fold0_best.pth
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
from scipy import stats

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
    DATASET_PATH, RESULTS_DIR, PATCH_SIZE, PATCH_OVERLAP,
    RANDOM_SEED, get_device, ensure_directories
)
from src.dataset import get_patient_folders, get_subject_splits, load_patient, get_file_paths
from src.models import build_model
from src.metrics import compute_patient_metrics, aggregate_metrics, print_metrics_table
from src.utils import set_seed, load_checkpoint, sliding_window_inference


def parse_args():
    parser = argparse.ArgumentParser(
        description="BraTS 2023 3D Segmentation Evaluation Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--model_variant", type=str, default="hybrid",
                        choices=["baseline", "channel", "spatial", "hybrid"],
                        help="Model architecture variant")
    parser.add_argument("--dataset_path", type=str, default=DATASET_PATH,
                        help="Path to dataset directory")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"],
                        help="Split to evaluate on")
    parser.add_argument("--fold", type=int, default=0,
                        help="K-Fold split index")
    parser.add_argument("--n_folds", type=int, default=3,
                        help="Total K-Fold partitions")
    parser.add_argument("--max_patients", type=int, default=None,
                        help="Subset of patients to evaluate")
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR,
                        help="Directory to save CSV tables and figures")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Custom CSV output filename")
    parser.add_argument("--compare_baseline", type=str, default=None,
                        help="Path to baseline checkpoint for paired Wilcoxon significance test")
    parser.add_argument("--save_nifti", action="store_true", default=False,
                        help="Save predicted segmentation masks as NIfTI (.nii.gz) files")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help="Random seed for consistent test split partitioning")
    return parser.parse_args()


def evaluate_cohort(model: torch.nn.Module,
                    patient_folders: List[str],
                    device: torch.device,
                    save_nifti_dir: str = None
                    ) -> List[Dict[str, Dict[str, float]]]:
    """Runs sliding-window inference and metric calculation over a cohort of patient scans."""
    model.eval()
    patient_results: List[Dict[str, Dict[str, float]]] = []

    for folder in tqdm(patient_folders, desc="Evaluating Scans"):
        patient_id = Path(folder).name
        image, gt_seg = load_patient(folder, has_seg=True)
        if gt_seg is None:
            continue

        # Get physical voxel spacing from NIfTI header
        paths = get_file_paths(folder)
        nii = nib.load(paths["flair"])
        voxel_spacing = tuple(abs(float(z)) for z in nii.header.get_zooms()[:3])

        img_tensor = torch.from_numpy(image.astype(np.float32))

        pred_seg = sliding_window_inference(
            model=model,
            volume=img_tensor,
            patch_size=PATCH_SIZE,
            overlap=PATCH_OVERLAP,
            device=device
        )

        metrics = compute_patient_metrics(pred_seg, gt_seg, voxel_spacing=voxel_spacing)
        metrics["_meta"] = {"patient_id": patient_id}
        patient_results.append(metrics)

        # Optionally export prediction as NIfTI
        if save_nifti_dir:
            out_nii_path = os.path.join(save_nifti_dir, f"{patient_id}_pred_seg.nii.gz")
            pred_nii = nib.Nifti1Image(pred_seg.astype(np.int16), affine=nii.affine, header=nii.header)
            nib.save(pred_nii, out_nii_path)

    return patient_results


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_directories([args.results_dir])
    device = get_device()

    print("\n" + "=" * 70)
    print("  📊 3D Brain Tumor Segmentation — Evaluation & Comparison")
    print(f"  Checkpoint    : {args.checkpoint}")
    print(f"  Model Variant : {args.model_variant.upper()}")
    print(f"  Target Split  : {args.split.upper()} (Fold {args.fold}/{args.n_folds})")
    print("=" * 70 + "\n")

    # 1. Discover patients and generate exact Subject-Grouped split
    folders = get_patient_folders(args.dataset_path)
    if args.max_patients and args.max_patients < len(folders):
        folders = folders[:args.max_patients]

    train_folders, val_folders, test_folders = get_subject_splits(
        folders, n_folds=args.n_folds, fold=args.fold, seed=args.seed
    )
    eval_folders = test_folders if args.split == "test" else val_folders

    print(f"  Evaluating {len(eval_folders)} patient scans on held-out {args.split.upper()} set.")

    # 2. Load Model
    model = build_model(variant=args.model_variant).to(device)
    load_checkpoint(model=model, checkpoint_path=args.checkpoint, device=device)

    # 3. Run Inference & Compute Metrics
    save_nii_dir = os.path.join(args.results_dir, "predictions") if args.save_nifti else None
    if save_nii_dir:
        os.makedirs(save_nii_dir, exist_ok=True)

    results = evaluate_cohort(model, eval_folders, device, save_nifti_dir=save_nii_dir)
    agg = aggregate_metrics(results)
    print_metrics_table(agg, model_name=f"{args.model_variant.upper()} ({Path(args.checkpoint).stem})")

    # 4. Save CSV Table
    csv_rows = []
    for r in results:
        pid = r["_meta"]["patient_id"]
        row = {"Patient_ID": pid}
        for region in ["WT", "TC", "ET"]:
            for m in ["dice", "iou", "hd95", "sensitivity", "specificity"]:
                row[f"{m.upper()}_{region}"] = r[region][m]
        csv_rows.append(row)

    df_patients = pd.DataFrame(csv_rows)
    csv_filename = args.output_csv or f"metrics_{args.model_variant}_{args.split}_fold{args.fold}.csv"
    csv_path = os.path.join(args.results_dir, csv_filename)
    df_patients.to_csv(csv_path, index=False)
    print(f"📁 Per-patient metrics exported to: {csv_path}")

    # Summary table
    summary_rows = []
    for region in ["WT", "TC", "ET"]:
        for m in ["dice", "iou", "hd95", "sensitivity", "specificity"]:
            stats_dict = agg[region][m]
            summary_rows.append({
                "Region": region,
                "Metric": m.upper(),
                "Mean":   round(stats_dict["mean"], 4),
                "Std":    round(stats_dict["std"], 4),
                "Median": round(stats_dict["median"], 4),
            })
    df_summary = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(args.results_dir, f"summary_{args.model_variant}_{args.split}_fold{args.fold}.csv")
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"📁 Cohort summary exported to: {summary_csv_path}")

    # 5. Optional Wilcoxon Signed-Rank Test vs Baseline
    if args.compare_baseline and os.path.exists(args.compare_baseline):
        print("\n" + "─" * 70)
        print("  🔬 Paired Statistical Significance Analysis (Wilcoxon Signed-Rank)")
        print(f"  Baseline Checkpoint: {args.compare_baseline}")
        print("─" * 70)

        baseline_model = build_model(variant="baseline").to(device)
        load_checkpoint(model=baseline_model, checkpoint_path=args.compare_baseline, device=device)
        baseline_results = evaluate_cohort(baseline_model, eval_folders, device)
        agg_bl = aggregate_metrics(baseline_results)

        for region in ["WT", "TC", "ET"]:
            scores_ours = [r[region]["dice"] for r in results if not np.isnan(r[region]["dice"])]
            scores_bl   = [r[region]["dice"] for r in baseline_results if not np.isnan(r[region]["dice"])]

            if len(scores_ours) > 0 and len(scores_ours) == len(scores_bl):
                stat, p_val = stats.wilcoxon(scores_ours, scores_bl, alternative="greater")
                sig = "✅ Statistically Significant (p < 0.05)" if p_val < 0.05 else "❌ Not Significant (p >= 0.05)"
                print(f"  {region} Dice: Baseline = {np.mean(scores_bl):.4f} → Proposed = {np.mean(scores_ours):.4f} "
                      f"| p-value = {p_val:.4e} | {sig}")

    print("\n✅ Evaluation complete.\n")


if __name__ == "__main__":
    main()
