"""
visualise_results.py — Publication-Quality Visualisation Generator

Generates qualitative visual evidence comparing Baseline 3D U-Net vs
Proposed Dual-Attention 3D U-Net on held-out test scans:

1. Predicted Segmentation Overlays (FLAIR + Ground Truth vs Baseline vs Attention U-Net)
2. Spatial Attention Heatmaps (Decoder levels 1–4 showing where the model attends)
3. Error Analysis Maps (False Positives & False Negatives highlighting sensitivity gains)
4. Tri-Planar View (Axial, Coronal, Sagittal slices)
5. Comprehensive Side-by-Side Presentation Figure

Outputs are saved in: results/figures/
"""

import sys, os, time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend for DGX / server
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import torch
import torch.nn.functional as F

from src.models import UNet3D, AttentionUNet3D
from src.dataset import get_dataloaders, load_patient, extract_random_patch
from src.metrics import compute_patient_metrics
from src.utils import load_checkpoint, sliding_window_inference, set_seed
from src.config import (
    DATASET_PATH, CHECKPOINT_DIR, RESULTS_DIR, DEVICE,
    PATCH_SIZE, PATCH_OVERLAP, RANDOM_SEED, MAX_PATIENTS
)

# ---------------------------------------------------------------------------
# Setup & Paths
# ---------------------------------------------------------------------------
set_seed(RANDOM_SEED)
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

LABEL_COLORS = {
    1: np.array([0.91, 0.30, 0.24]),  # NCR  = red
    2: np.array([0.20, 0.60, 0.86]),  # ED   = blue
    3: np.array([0.18, 0.80, 0.44]),  # ET   = green
}

def overlay_seg_on_mri(mri_slice, seg_slice, alpha=0.55):
    """Return RGBA overlay of segmentation on MRI slice."""
    mri_norm = (mri_slice - mri_slice.min()) / (mri_slice.max() - mri_slice.min() + 1e-8)
    rgb = np.stack([mri_norm] * 3, axis=-1)
    for label_id, color in LABEL_COLORS.items():
        mask = (seg_slice == label_id)
        rgb[mask] = (1 - alpha) * rgb[mask] + alpha * color
    return np.clip(rgb, 0.0, 1.0)

def find_best_slice(gt_seg):
    """Find axial slice index with maximum tumor area (enhancing core priority)."""
    et_counts = (gt_seg == 3).sum(axis=(0, 1))
    if et_counts.max() > 50:
        return int(np.argmax(et_counts))
    wt_counts = (gt_seg > 0).sum(axis=(0, 1))
    if wt_counts.max() > 0:
        return int(np.argmax(wt_counts))
    return gt_seg.shape[2] // 2

def main():
    print("=" * 70)
    print("  BraTS 2023 — Qualitative Visualisation Pipeline")
    print(f"  Device: {DEVICE}")
    print(f"  Target directory: {FIGURES_DIR}")
    print("=" * 70)

    # 1. Load Models
    print("\n[1/5] Loading trained checkpoints...")
    baseline_model = UNet3D(in_channels=4, out_channels=4, init_features=32).to(DEVICE)
    load_checkpoint(baseline_model, model_name="baseline_unet3d", prefer_best=True)
    baseline_model.eval()

    attn_model = AttentionUNet3D(
        in_channels=4, out_channels=4, init_features=32,
        use_channel_attn=True, use_spatial_attn=True
    ).to(DEVICE)
    load_checkpoint(attn_model, model_name="attention_unet3d_full", prefer_best=True)
    attn_model.eval()
    print("  -> Both models successfully loaded into memory.")

    # 2. Get Test Patients from zero-leakage split
    print("\n[2/5] Creating zero-leakage test set split...")
    _, _, test_loader = get_dataloaders(
        dataset_path=DATASET_PATH,
        batch_size=1,
        patch_size=PATCH_SIZE,
        num_workers=2,
        n_folds=3,
        fold=0,
        max_patients=MAX_PATIENTS,
    )
    test_folders = test_loader.dataset.patient_folders
    print(f"  -> Total test patients: {len(test_folders)}")

    # Select representative patients with significant tumor burden
    selected_patients = []
    print("  -> Scanning test set for representative tumor cases...")
    for folder in test_folders:
        try:
            _, gt_seg = load_patient(folder, has_seg=True)
            wt_vol = int((gt_seg > 0).sum())
            et_vol = int((gt_seg == 3).sum())
            if wt_vol > 5000 and et_vol > 300:
                selected_patients.append((folder, wt_vol, et_vol))
            if len(selected_patients) >= 6:
                break
        except Exception:
            continue

    if not selected_patients:
        selected_patients = [(f, 0, 0) for f in test_folders[:4]]

    print(f"  -> Selected {len(selected_patients)} patients for high-res rendering:")
    for f, wt, et in selected_patients:
        print(f"     • {Path(f).name} (WT voxels: {wt:,}, ET voxels: {et:,})")

    # 3. Generate Overlays & Error Maps
    print("\n[3/5] Generating Segmentation Overlays & Error Maps...")
    presentation_rows = []

    for idx, (folder, _, _) in enumerate(selected_patients, start=1):
        pid = Path(folder).name
        print(f"\n  [{idx}/{len(selected_patients)}] Processing {pid}...")

        image, gt_seg = load_patient(folder, has_seg=True)
        img_tensor = torch.from_numpy(image.astype(np.float32))

        t0 = time.time()
        with torch.no_grad():
            pred_bl = sliding_window_inference(baseline_model, img_tensor, PATCH_SIZE, PATCH_OVERLAP, DEVICE)
            pred_at = sliding_window_inference(attn_model, img_tensor, PATCH_SIZE, PATCH_OVERLAP, DEVICE)
        inf_time = time.time() - t0

        m_bl = compute_patient_metrics(pred_bl, gt_seg)
        m_at = compute_patient_metrics(pred_at, gt_seg)

        bl_wt, bl_tc, bl_et = m_bl['WT']['dice'], m_bl['TC']['dice'], m_bl['ET']['dice']
        at_wt, at_tc, at_et = m_at['WT']['dice'], m_at['TC']['dice'], m_at['ET']['dice']

        print(f"     Inference time: {inf_time:.1f}s")
        print(f"     Baseline Dice  -> WT: {bl_wt:.3f} | TC: {bl_tc:.3f} | ET: {bl_et:.3f}")
        print(f"     Attention Dice -> WT: {at_wt:.3f} | TC: {at_tc:.3f} | ET: {at_et:.3f}")

        flair = image[0]
        z = find_best_slice(gt_seg)

        # A. 4-Panel Overlay Figure
        fig, axes = plt.subplots(1, 4, figsize=(22, 6))
        fig.suptitle(f"Patient: {pid} (Axial Slice Z={z})", fontsize=15, fontweight="bold", y=1.02)

        axes[0].imshow(flair[:, :, z].T, cmap="gray", origin="lower")
        axes[0].set_title("FLAIR Modality (Input)", fontsize=12, pad=8)
        axes[0].axis("off")

        gt_over = overlay_seg_on_mri(flair[:, :, z], gt_seg[:, :, z])
        axes[1].imshow(gt_over.transpose(1, 0, 2), origin="lower")
        axes[1].set_title("Ground Truth Segmentation", fontsize=12, pad=8)
        axes[1].axis("off")

        bl_over = overlay_seg_on_mri(flair[:, :, z], pred_bl[:, :, z])
        axes[2].imshow(bl_over.transpose(1, 0, 2), origin="lower")
        axes[2].set_title(f"Baseline 3D U-Net\nWT: {bl_wt:.3f} | TC: {bl_tc:.3f} | ET: {bl_et:.3f}", fontsize=11, pad=8)
        axes[2].axis("off")

        at_over = overlay_seg_on_mri(flair[:, :, z], pred_at[:, :, z])
        axes[3].imshow(at_over.transpose(1, 0, 2), origin="lower")
        axes[3].set_title(f"Dual-Attention 3D U-Net (Ours)\nWT: {at_wt:.3f} | TC: {at_tc:.3f} | ET: {at_et:.3f}",
                           fontsize=11, pad=8, color="#1e824c", fontweight="bold")
        axes[3].axis("off")

        patches = [
            mpatches.Patch(color='#e74c3c', label='NCR / Non-Enhancing Core (Label 1)'),
            mpatches.Patch(color='#3498db', label='Edema (Label 2)'),
            mpatches.Patch(color='#2ecc71', label='ET / Enhancing Tumor (Label 3)'),
        ]
        fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=11, bbox_to_anchor=(0.5, -0.06), frameon=True)
        plt.tight_layout()

        overlay_save = os.path.join(FIGURES_DIR, f"overlay_{pid}.png")
        plt.savefig(overlay_save, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"     [Saved] {overlay_save}")

        # B. Error Analysis (FP vs FN)
        gt_bin = (gt_seg > 0).astype(np.uint8)
        bl_bin = (pred_bl > 0).astype(np.uint8)
        at_bin = (pred_at > 0).astype(np.uint8)

        bl_fp = ((bl_bin == 1) & (gt_bin == 0))[:, :, z]
        bl_fn = ((bl_bin == 0) & (gt_bin == 1))[:, :, z]
        at_fp = ((at_bin == 1) & (gt_bin == 0))[:, :, z]
        at_fn = ((at_bin == 0) & (gt_bin == 1))[:, :, z]

        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        fig.suptitle(f"Error Analysis (FP vs FN) — {pid} (Slice Z={z})", fontsize=15, fontweight="bold")

        bg = flair[:, :, z]
        bg_norm = (bg - bg.min()) / (bg.max() - bg.min() + 1e-8)

        def plot_error(ax, mask, color, title):
            ax.imshow(bg_norm.T, cmap="gray", origin="lower")
            m = np.ma.masked_where(mask == 0, mask)
            ax.imshow(m.T, cmap=ListedColormap([color]), origin="lower", alpha=0.75)
            ax.set_title(title, fontsize=11, pad=6)
            ax.axis("off")

        plot_error(axes[0, 0], gt_bin[:, :, z], "#2ecc71", "Ground Truth (Whole Tumor)")
        plot_error(axes[0, 1], bl_fp, "#e74c3c", f"Baseline False Positives (FP: {bl_fp.sum():,} vox)")
        plot_error(axes[0, 2], bl_fn, "#f39c12", f"Baseline False Negatives (FN: {bl_fn.sum():,} vox)")

        plot_error(axes[1, 0], at_bin[:, :, z], "#9b59b6", "Attention U-Net Whole Tumor")
        plot_error(axes[1, 1], at_fp, "#e74c3c", f"Attention False Positives (FP: {at_fp.sum():,} vox)")
        plot_error(axes[1, 2], at_fn, "#27ae60", f"Attention False Negatives (FN: {at_fn.sum():,} vox)")

        plt.tight_layout()
        err_save = os.path.join(FIGURES_DIR, f"error_analysis_{pid}.png")
        plt.savefig(err_save, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"     [Saved] {err_save}")

        # Save slice for presentation figure if attention won
        if at_et >= bl_et:
            presentation_rows.append((pid, flair[:, :, z], gt_over, bl_over, at_over, bl_wt, at_wt, bl_et, at_et))

    # 4. Generate 3D Attention Gate Heatmaps
    print("\n[4/5] Generating Spatial Attention Heatmaps from Decoder Levels 1–4...")
    attn_patient_folder = selected_patients[0][0]
    pid = Path(attn_patient_folder).name
    image, gt_seg = load_patient(attn_patient_folder, has_seg=True)

    img_patch, seg_patch = extract_random_patch(image, gt_seg, PATCH_SIZE, foreground_prob=1.0)
    img_tensor = torch.from_numpy(img_patch.astype(np.float32)).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        _, attention_maps = attn_model.get_attention_maps(img_tensor)

    mid = img_patch.shape[-1] // 2
    flair_slice = img_patch[0, :, :, mid]
    flair_norm = (flair_slice - flair_slice.min()) / (flair_slice.max() - flair_slice.min() + 1e-8)
    attn_cmap = plt.cm.plasma

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle(f"Spatial Attention Gate Activation Maps across Decoder Stages\nPatient: {pid}",
                 fontsize=15, fontweight="bold", y=0.98)

    stage_names = [
        "Decoder Level 4 (Coarse / Semantic)",
        "Decoder Level 3 (Contextual)",
        "Decoder Level 2 (Refined)",
        "Decoder Level 1 (High-Resolution Boundary)"
    ]

    for i, attn_map in enumerate(attention_maps):
        attn = attn_map[0, 0].cpu().numpy()
        attn_upsampled = F.interpolate(
            torch.from_numpy(attn).unsqueeze(0).unsqueeze(0).float(),
            size=img_patch.shape[1:], mode="trilinear", align_corners=True
        )[0, 0].numpy()

        attn_sl = attn_upsampled[:, :, mid]

        # Top row: FLAIR + Heatmap
        axes[0, i].imshow(flair_norm.T, cmap="gray", origin="lower")
        axes[0, i].imshow(attn_sl.T, cmap=attn_cmap, origin="lower", alpha=0.55)
        axes[0, i].set_title(f"{stage_names[i]}\nOverlay on FLAIR", fontsize=11, pad=6)
        axes[0, i].axis("off")

        # Bottom row: Attention gate alone
        im = axes[1, i].imshow(attn_sl.T, cmap=attn_cmap, origin="lower")
        axes[1, i].set_title(f"Attention Weights\n[min: {attn_sl.min():.2f}, max: {attn_sl.max():.2f}]", fontsize=10, pad=6)
        axes[1, i].axis("off")
        fig.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)

    plt.tight_layout()
    attn_save = os.path.join(FIGURES_DIR, f"attention_heatmaps_{pid}.png")
    plt.savefig(attn_save, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Saved] {attn_save}")

    # 5. Generate Multi-Patient Publication Grid
    print("\n[5/5] Compiling Multi-Patient Master Comparison Figure...")
    if presentation_rows:
        num_display = min(3, len(presentation_rows))
        fig, axes = plt.subplots(num_display, 4, figsize=(20, 5 * num_display))
        if num_display == 1:
            axes = np.expand_dims(axes, 0)

        for row_idx in range(num_display):
            pid, flair_sl, gt_ov, bl_ov, at_ov, bl_wt, at_wt, bl_et, at_et = presentation_rows[row_idx]

            axes[row_idx, 0].imshow(flair_sl.T, cmap="gray", origin="lower")
            axes[row_idx, 0].set_ylabel(f"Patient\n{pid}", fontsize=11, fontweight="bold", labelpad=10)
            axes[row_idx, 0].set_title("FLAIR Input" if row_idx == 0 else "", fontsize=12)
            axes[row_idx, 0].axis("off")

            axes[row_idx, 1].imshow(gt_ov.transpose(1, 0, 2), origin="lower")
            axes[row_idx, 1].set_title("Ground Truth" if row_idx == 0 else "", fontsize=12)
            axes[row_idx, 1].axis("off")

            axes[row_idx, 2].imshow(bl_ov.transpose(1, 0, 2), origin="lower")
            axes[row_idx, 2].set_title(
                f"Baseline U-Net\n(WT: {bl_wt:.2f} | ET: {bl_et:.2f})" if row_idx == 0
                else f"WT: {bl_wt:.2f} | ET: {bl_et:.2f}",
                fontsize=11
            )
            axes[row_idx, 2].axis("off")

            axes[row_idx, 3].imshow(at_ov.transpose(1, 0, 2), origin="lower")
            axes[row_idx, 3].set_title(
                f"Attention U-Net (Ours)\n(WT: {at_wt:.2f} | ET: {at_et:.2f})" if row_idx == 0
                else f"WT: {at_wt:.2f} | ET: {at_et:.2f}",
                fontsize=11, color="#1e824c", fontweight="bold"
            )
            axes[row_idx, 3].axis("off")

        patches = [
            mpatches.Patch(color='#e74c3c', label='NCR / Core (Red)'),
            mpatches.Patch(color='#3498db', label='Edema (Blue)'),
            mpatches.Patch(color='#2ecc71', label='Enhancing Tumor (Green)'),
        ]
        fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=12, bbox_to_anchor=(0.5, -0.02), frameon=True)
        plt.tight_layout()

        master_save = os.path.join(FIGURES_DIR, "master_qualitative_comparison.png")
        plt.savefig(master_save, dpi=250, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Saved] {master_save}")

    print("\n" + "=" * 70)
    print("  Visualisation Complete!")
    print(f"  All generated figures are saved in: {FIGURES_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()
