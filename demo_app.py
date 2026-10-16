"""
demo_app.py — Interactive Brain Tumor Segmentation & Ablation Dashboard
========================================================================
Run with:   streamlit run demo_app.py

Features:
  1. Interactive 3D MRI Slice Viewer (Loads built-in sample patient or upload custom NIfTI)
  2. NVIDIA B200 Ablation Study & Quantitative Benchmarks (Live CSV metrics + Charts)
  3. Qualitative Test Gallery & Attention Interpretability (Master comparisons, overlays, error analysis)
  4. Proposed Dual-Attention Architecture & DGX Specifications
"""

import io
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import tempfile
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    import nibabel as nib
    NIBABEL_OK = True
except ImportError:
    NIBABEL_OK = False

try:
    import torch
    TORCH_OK = True
except ImportError:
    TORCH_OK = False

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor Segmentation | Dual-Attention 3D U-Net",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Modern Dark Medical Theme CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }

    .main { background-color: #0b0f19; }

    .hero-banner {
        background: linear-gradient(135deg, #0d1b2a 0%, #1b263b 40%, #1d3557 100%);
        border: 1px solid #2a3d5e;
        border-radius: 16px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }
    .hero-title {
        color: #64dfdf;
        font-size: 2.3rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-sub {
        color: #e0e1dd;
        font-size: 1.1rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    .hero-meta {
        color: #90e0ef;
        font-size: 0.88rem;
        margin-top: 0.8rem;
        font-weight: 500;
    }

    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 1.1rem 1.2rem;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: #3b82f6;
    }
    .kpi-val {
        font-size: 2.1rem;
        font-weight: 800;
        color: #10b981;
        line-height: 1.1;
    }
    .kpi-title {
        font-size: 0.82rem;
        color: #9ca3af;
        margin-top: 0.4rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-delta {
        font-size: 0.78rem;
        color: #38bdf8;
        margin-top: 0.2rem;
        font-weight: 600;
    }

    .info-card {
        background: #111827;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        color: #d1d5db;
    }

    .badge-ncr { background: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid #ef4444; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }
    .badge-ed  { background: rgba(59, 130, 246, 0.2); color: #3b82f6; border: 1px solid #3b82f6; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }
    .badge-et  { background: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid #10b981; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }

    div[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CONSTANTS & COLOUR MAP
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.resolve()
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
SAMPLE_PATIENT = "BraTS-GLI-00659-000"

# BraTS 2023 Labels
LABEL_NAMES = {
    0: "Background",
    1: "Necrotic Core (NCR)",
    2: "Peritumoral Edema (ED)",
    3: "Enhancing Tumor (ET)",
}

# Segmentation RGBA colors (RGB + Alpha)
SEG_COLORS = {
    0: (0,   0,   0,   0  ),   # Background — transparent
    1: (239, 68,  68,  190),   # NCR — Vibrant Red
    2: (59,  130, 246, 170),   # ED  — Vibrant Blue
    3: (16,  185, 129, 210),   # ET  — Vibrant Green
}


# =============================================================================
# CACHED DATA & IMAGE LOADERS
# =============================================================================
@st.cache_data(show_spinner="Loading NIfTI volume...")
def load_nifti_from_path(path: str) -> np.ndarray:
    """Load NIfTI image from disk."""
    img = nib.load(path)
    return img.get_fdata().astype(np.float32)

@st.cache_data(show_spinner="Processing uploaded NIfTI...")
def load_nifti_from_bytes(file_bytes: bytes) -> np.ndarray:
    """Load NIfTI image from uploaded file buffer."""
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
        f.write(file_bytes)
        tmp = f.name
    img = nib.load(tmp)
    vol = img.get_fdata().astype(np.float32)
    try:
        os.remove(tmp)
    except OSError:
        pass
    return vol

def seg_to_rgba(seg_slice: np.ndarray) -> np.ndarray:
    """Convert 2D integer label slice into RGBA color overlay."""
    h, w = seg_slice.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for label, color in SEG_COLORS.items():
        mask = (seg_slice == label)
        if mask.any():
            rgba[mask] = color
    return rgba

def normalize_slice(s: np.ndarray) -> np.ndarray:
    """Normalize 2D MRI slice to [0, 1] range."""
    mn, mx = s.min(), s.max()
    if mx == mn:
        return np.zeros_like(s, dtype=np.float32)
    return ((s - mn) / (mx - mn)).astype(np.float32)

def make_overlay_figure(mri_slice: np.ndarray, seg_slice: np.ndarray,
                        modality_name: str = "FLAIR", title: str = "",
                        alpha: float = 0.50):
    """Render 3-panel display: Raw MRI, Segmentation Mask, Alpha Overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), facecolor="#0b0f19",
                             gridspec_kw={"wspace": 0.04})

    mri_norm = normalize_slice(mri_slice)
    rgba = seg_to_rgba(seg_slice)

    panels = [
        (mri_norm, f"MRI ({modality_name})", "gray"),
        (np.zeros_like(mri_norm), "Segmentation Map", "gray"),
        (mri_norm, "Clinical Overlay", "gray"),
    ]

    for ax, (img, lbl, cmap) in zip(axes, panels):
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        if lbl == "Segmentation Map":
            ax.imshow(rgba)
        elif lbl == "Clinical Overlay":
            ax.imshow(rgba, alpha=alpha)
        ax.set_title(lbl, color="#e0e1dd", fontsize=11, fontweight="600", pad=6)
        ax.axis("off")
        ax.set_facecolor("#0b0f19")

    if title:
        fig.suptitle(title, color="#64dfdf", fontsize=13, fontweight="700", y=1.01)

    patches = [
        mpatches.Patch(color=np.array(SEG_COLORS[1][:3])/255, label="Necrotic Core (NCR)"),
        mpatches.Patch(color=np.array(SEG_COLORS[2][:3])/255, label="Peritumoral Edema (ED)"),
        mpatches.Patch(color=np.array(SEG_COLORS[3][:3])/255, label="Enhancing Tumor (ET)"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               facecolor="#111827", edgecolor="#1f2937",
               labelcolor="#e0e1dd", fontsize=9.5, framealpha=1,
               bbox_to_anchor=(0.5, -0.09))
    return fig


# =============================================================================
# HEADER / HERO SECTION
# =============================================================================
st.markdown("""
<div class="hero-banner">
  <div class="hero-title">🧠 Brain Tumor Segmentation Dashboard</div>
  <div class="hero-sub">Dual-Attention 3D U-Net (SE Channel + Spatial Gates) · BraTS 2023 · NVIDIA B200 DGX</div>
  <div class="hero-meta">
    👥 <b>Team:</b> Rishi Agrawal · Rishil Pawar · Sahil Deotale · Shrishti Lal &nbsp;|&nbsp;
    🎓 <b>Guide:</b> Dr. Uma Yadav &nbsp;|&nbsp;
    🏫 <b>RCOEM Nagpur (CSE - Data Science)</b>
  </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Navigation & Controls")
    app_mode = st.radio(
        "Select View:",
        [
            "🔬 3D MRI Slice Viewer",
            "📊 Ablation Study & Benchmarks",
            "🖼️ Clinical Test Gallery",
            "🏗️ Architecture & DGX Specs",
        ],
        index=0,
    )
    st.markdown("---")

    if app_mode == "🔬 3D MRI Slice Viewer":
        st.markdown("#### 📁 Scan Input Source")
        data_source = st.radio(
            "Choose MRI Data:",
            ["Use Preloaded Patient (BraTS-GLI-00659-000)", "Upload Custom .nii.gz Files"],
            index=0,
        )
        overlay_alpha = st.slider("Overlay Opacity", 0.1, 0.9, 0.50, 0.05)
        view_plane = st.selectbox("Anatomical Plane", ["Axial (Transverse)", "Coronal (Frontal)", "Sagittal (Side)"], index=0)
    else:
        st.markdown("#### 💡 Quick Summary")
        st.markdown("""
        - **Target Dice:** ~0.87 WT
        - **Achieved WT Dice:** **0.9228** (+5.28%)
        - **ET Sensitivity:** **86.64%** (+3.20%)
        - **Parameter Overhead:** **Only +0.62%**
        - **Hardware:** NVIDIA Blackwell B200 MIG
        """)

    st.markdown("---")
    st.caption("RCOEM B.Tech CSE-DS (2026–2027) | Semester VII Project")


# =============================================================================
# TAB 1: 🔬 INTERACTIVE 3D MRI SLICE VIEWER
# =============================================================================
if app_mode == "🔬 3D MRI Slice Viewer":
    st.subheader("🔬 Volumetric 3D MRI Slice & Segmentation Inspector")

    flair_vol, seg_vol = None, None
    patient_id_display = ""
    available_modalities = {}

    if data_source == "Use Preloaded Patient (BraTS-GLI-00659-000)":
        patient_id_display = SAMPLE_PATIENT
        flair_path = PROJECT_ROOT / f"{SAMPLE_PATIENT}-t2f.nii.gz"
        seg_path   = PROJECT_ROOT / f"{SAMPLE_PATIENT}-seg.nii.gz"

        if flair_path.exists() and seg_path.exists():
            flair_vol = load_nifti_from_path(str(flair_path))
            seg_vol   = load_nifti_from_path(str(seg_path)).astype(np.int32)
            available_modalities["FLAIR"] = flair_vol

            # Also load other modalities if present in root
            for mod_key, mod_file in [("T1ce", f"{SAMPLE_PATIENT}-t1c.nii.gz"),
                                      ("T1",   f"{SAMPLE_PATIENT}-t1n.nii.gz"),
                                      ("T2",   f"{SAMPLE_PATIENT}-t2w.nii.gz")]:
                m_path = PROJECT_ROOT / mod_file
                if m_path.exists():
                    available_modalities[mod_key] = load_nifti_from_path(str(m_path))
        else:
            st.warning(f"Preloaded files not found at {flair_path}. Please switch to upload mode.")
    else:
        # Custom upload mode
        c1, c2 = st.columns(2)
        with c1:
            u_flair = st.file_uploader("Upload FLAIR / Structural MRI (.nii / .nii.gz)", type=["nii", "gz"], key="u_flair")
        with c2:
            u_seg   = st.file_uploader("Upload Segmentation Mask (.nii / .nii.gz)", type=["nii", "gz"], key="u_seg")

        if u_flair and u_seg:
            flair_vol = load_nifti_from_bytes(u_flair.read())
            seg_vol   = load_nifti_from_bytes(u_seg.read()).astype(np.int32)
            available_modalities["Uploaded MRI"] = flair_vol
            patient_id_display = u_flair.name.split("-")[0] if "-" in u_flair.name else "Custom Scan"

    if flair_vol is not None and seg_vol is not None:
        # Check volume plane orientation
        if view_plane.startswith("Axial"):
            # Slices along Z axis (dim 2)
            vol_mri = flair_vol
            vol_seg = seg_vol
            total_slices = vol_mri.shape[2]
            slice_fn = lambda idx: (vol_mri[:, :, idx].T, vol_seg[:, :, idx].T)
            plane_desc = "Axial (Inferior ↔ Superior)"
        elif view_plane.startswith("Coronal"):
            vol_mri = flair_vol
            vol_seg = seg_vol
            total_slices = vol_mri.shape[1]
            slice_fn = lambda idx: (vol_mri[:, idx, :].T, vol_seg[:, idx, :].T)
            plane_desc = "Coronal (Anterior ↔ Posterior)"
        else:
            vol_mri = flair_vol
            vol_seg = seg_vol
            total_slices = vol_mri.shape[0]
            slice_fn = lambda idx: (vol_mri[idx, :, :].T, vol_seg[idx, :, :].T)
            plane_desc = "Sagittal (Left ↔ Right)"

        # Find slice with maximum tumor burden
        tumor_counts = [(vol_seg[:, :, i] > 0).sum() for i in range(vol_seg.shape[2])] if view_plane.startswith("Axial") else [(vol_seg > 0).sum()]
        default_slice = int(np.argmax(tumor_counts)) if len(tumor_counts) > 1 else total_slices // 2

        # Top control row
        r1, r2, r3 = st.columns([2, 1, 1])
        with r1:
            slice_idx = st.slider(f"Browse {plane_desc} Slices", 0, total_slices - 1, default_slice, 1)
        with r2:
            chosen_mod = st.selectbox("MRI Contrast Channel", list(available_modalities.keys()), index=0)
        with r3:
            total_tumor_voxels = int((seg_vol > 0).sum())
            st.metric("Total 3D Tumor Volume", f"{total_tumor_voxels:,} voxels", help="1 voxel = 1.0 mm³ isotropic")

        # Modality selection
        curr_mri_vol = available_modalities.get(chosen_mod, flair_vol)
        if view_plane.startswith("Axial"):
            mri_slice, seg_slice = curr_mri_vol[:, :, slice_idx].T, seg_vol[:, :, slice_idx].T
        elif view_plane.startswith("Coronal"):
            mri_slice, seg_slice = curr_mri_vol[:, slice_idx, :].T, seg_vol[:, slice_idx, :].T
        else:
            mri_slice, seg_slice = curr_mri_vol[slice_idx, :, :].T, seg_vol[slice_idx, :, :].T

        # Render Figure
        fig = make_overlay_figure(
            mri_slice, seg_slice,
            modality_name=chosen_mod,
            title=f"Patient: {patient_id_display} — {plane_desc} Slice {slice_idx} / {total_slices - 1}",
            alpha=overlay_alpha,
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Slice Breakdown Metrics
        st.markdown("#### 📊 Slice Quantitative Breakdown")
        m1, m2, m3, m4 = st.columns(4)
        ncr_count = int((seg_slice == 1).sum())
        ed_count  = int((seg_slice == 2).sum())
        et_count  = int((seg_slice == 3).sum())
        wt_count  = ncr_count + ed_count + et_count

        m1.metric("Whole Tumor (WT)", f"{wt_count:,} px", "All Regions")
        m2.metric("Necrotic Core (NCR)", f"{ncr_count:,} px", "Label 1 (Red)")
        m3.metric("Peritumoral Edema (ED)", f"{ed_count:,} px", "Label 2 (Blue)")
        m4.metric("Enhancing Tumor (ET)", f"{et_count:,} px", "Label 3 (Green)")

        # Download button
        buf = io.BytesIO()
        fig_dl = make_overlay_figure(mri_slice, seg_slice, modality_name=chosen_mod, alpha=overlay_alpha)
        fig_dl.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="#0b0f19")
        plt.close(fig_dl)
        buf.seek(0)
        st.download_button(
            "⬇️ Download High-Res Slice Overlay (PNG)",
            data=buf,
            file_name=f"{patient_id_display}_slice_{slice_idx}.png",
            mime="image/png",
        )
    else:
        st.info("Please select or upload MRI scans to display the interactive 3D slice viewer.")


# =============================================================================
# TAB 2: 📊 B200 ABLATION STUDY & BENCHMARKS
# =============================================================================
elif app_mode == "📊 Ablation Study & Benchmarks":
    st.subheader("📊 NVIDIA B200 Overnight Ablation Study & Benchmarks")
    st.markdown("""
    Rigorous evaluation conducted on **122 held-out test patients** (Subject-Grouped, 0 patient leakage)
    isolating the incremental contribution of **Channel Attention (SE)** and **Spatial Attention Gates**.
    """)

    # Top KPI Metrics Cards
    st.markdown("""
    <div class="kpi-container">
      <div class="kpi-card">
        <div class="kpi-val">0.9228</div>
        <div class="kpi-title">Whole Tumor (WT) Dice</div>
        <div class="kpi-delta">+5.28% over 0.87 Target</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-val">0.8877</div>
        <div class="kpi-title">Tumor Core (TC) Dice</div>
        <div class="kpi-delta">Consistently ~89%</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-val">0.8517</div>
        <div class="kpi-title">Enhancing Tumor (ET) Dice</div>
        <div class="kpi-delta">High Precision Active Core</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-val" style="color: #38bdf8;">86.64%</div>
        <div class="kpi-title">ET Sensitivity (Recall)</div>
        <div class="kpi-delta">+3.20% over Baseline</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-val" style="color: #f59e0b;">+0.62%</div>
        <div class="kpi-title">Parameter Overhead</div>
        <div class="kpi-delta">22.72M vs 22.58M Params</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Load Ablation CSV
    ablation_csv_path = RESULTS_DIR / "ablation_study.csv"
    if ablation_csv_path.exists():
        st.markdown("#### 📋 Full Multi-Model Ablation Comparison Table")
        df_ablation = pd.read_csv(ablation_csv_path)

        # Style table
        st.dataframe(
            df_ablation.style.highlight_max(subset=["Dice WT", "Dice TC", "Dice ET", "Sens WT", "Sens TC", "Sens ET"], color="#1e3a8a")
                             .highlight_min(subset=["HD95 WT", "HD95 ET", "Params (M)"], color="#064e3b"),
            use_container_width=True,
            hide_index=True,
        )

        st.caption("🏆 **Highlighting:** Blue indicates maximum segmentation accuracy / sensitivity; Green indicates lowest boundary error (HD95) and model footprint.")
    else:
        st.warning("ablation_study.csv not found in results directory.")

    st.markdown("---")

    # Display Ablation Bar Charts and Boxplots
    col_plot1, col_plot2 = st.columns(2)
    with col_plot1:
        st.markdown("##### 📈 Ablation Study — Attention Module Comparison")
        ablation_plot = RESULTS_DIR / "ablation_barchart.png"
        if ablation_plot.exists():
            st.image(str(ablation_plot), caption="Ablation Bar Chart: Baseline vs Channel vs Spatial vs Proposed Dual Attention", use_container_width=True)
        else:
            st.info("Ablation bar chart pending.")

    with col_plot2:
        st.markdown("##### 📦 Dice Distribution & Variance Across Test Scans")
        box_plot = RESULTS_DIR / "dice_boxplots.png"
        if box_plot.exists():
            st.image(str(box_plot), caption="Boxplot of Dice Scores across WT, TC, and ET on held-out test cohort", use_container_width=True)
        else:
            dice_comp = RESULTS_DIR / "dice_comparison_barchart.png"
            if dice_comp.exists():
                st.image(str(dice_comp), caption="Dice comparison across tumor regions", use_container_width=True)

    # Clinical Analysis Note
    st.markdown("""
    <div class="info-card">
      <b>💡 Key Defense Finding — Enhancing Tumor (ET) Recall:</b><br>
      In clinical neuro-oncology, Enhancing Tumor (ET) represents the rapidly dividing, angiogenic cancerous core.
      Missing ET tissue (false negative) during surgical resection planning leads to early tumor recurrence.
      Our proposed <b>Dual-Attention U-Net boosted ET Sensitivity from 83.44% → 86.64% (+3.20% gain)</b>
      and TC Sensitivity from <b>87.80% → 90.48% (+2.68% gain)</b> with virtually zero parameter overhead (+0.62%).
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# TAB 3: 🖼️ CLINICAL TEST GALLERY & INTERPRETABILITY
# =============================================================================
elif app_mode == "🖼️ Clinical Test Gallery":
    st.subheader("🖼️ Clinical Qualitative Gallery & Model Interpretability")
    st.markdown("Visual validation of segmentations, boundary error analysis, and spatial attention activations.")

    # Master Qualitative Comparison
    master_comp = FIGURES_DIR / "master_qualitative_comparison.png"
    if master_comp.exists():
        st.markdown("#### 🌟 Master Multi-Model Comparison (Ground Truth vs All Variants)")
        st.image(str(master_comp), caption="Axial comparison across Ground Truth, Baseline 3D U-Net, Channel-Only, Spatial-Only, and Proposed Dual-Attention", use_container_width=True)
        st.markdown("---")

    # Patient Selector
    test_patients = [
        "BraTS-GLI-01279-000",
        "BraTS-GLI-00292-000",
        "BraTS-GLI-00651-000",
        "BraTS-GLI-00824-000",
        "BraTS-GLI-01106-000",
        "BraTS-GLI-01307-000",
    ]

    selected_pt = st.selectbox("Select Test Patient for In-Depth Clinical Inspection:", test_patients, index=0)

    g1, g2 = st.columns(2)
    with g1:
        st.markdown(f"##### 🎯 Model Delineation Overlay ({selected_pt})")
        overlay_fig = FIGURES_DIR / f"overlay_{selected_pt}.png"
        if overlay_fig.exists():
            st.image(str(overlay_fig), caption="MRI with Model Predicted Boundaries (Red=NCR, Blue=ED, Green=ET)", use_container_width=True)
        else:
            st.info("Overlay figure not found.")

    with g2:
        st.markdown(f"##### 🔍 Boundary Error Analysis ({selected_pt})")
        error_fig = FIGURES_DIR / f"error_analysis_{selected_pt}.png"
        if error_fig.exists():
            st.image(str(error_fig), caption="Voxel-level Error Map: Green = True Positive (TP), Red = False Positive (FP), Blue = False Negative (FN)", use_container_width=True)
        else:
            st.info("Error analysis figure not found.")

    st.markdown("---")

    # Attention Heatmaps
    st.markdown("#### 🧠 Spatial Attention Gate Interpretability (Feature Activation)")
    attn_fig = FIGURES_DIR / f"attention_heatmaps_{selected_pt}.png"
    if not attn_fig.exists():
        # Fallback to the available attention heatmap
        attn_fig = FIGURES_DIR / "attention_heatmaps_BraTS-GLI-01279-000.png"

    if attn_fig.exists():
        st.image(str(attn_fig), caption="Spatial Attention Gate activations across decoder skip connections: background brain tissue is attenuated while the active tumor margins are magnified.", use_container_width=True)
    else:
        st.info("Attention heatmaps not found in figures directory.")


# =============================================================================
# TAB 4: 🏗️ ARCHITECTURE & DGX SPECS
# =============================================================================
elif app_mode == "🏗️ Architecture & DGX Specs":
    st.subheader("🏗️ Proposed Dual-Attention Architecture & DGX Infrastructure")

    c_arch, c_dgx = st.columns(2)

    with c_arch:
        st.markdown("### 🧩 Dual-Attention Mechanism")
        st.markdown("""
        Standard 3D U-Nets treat every voxel and every feature channel equally, leading to blurry boundaries in heterogeneous tumors.
        Our proposed model integrates a **hybrid attention mechanism** directly onto every decoder skip connection:

        1. **Channel Attention (Squeeze-and-Excitation — SE):**
           - Compresses spatial dimensions via global average pooling.
           - Employs a two-layer MLP with reduction ratio $r=16$ and Sigmoid gating.
           - Dynamically recalibrates the importance of different MRI modality feature maps.

        2. **Spatial Attention Gates:**
           - Computes pixel-wise attention coefficients using coarse gating signals from the deeper decoder layer.
           - Suppresses background tissue and normal brain parenchyma while amplifying boundary gradients.

        3. **End-to-End Objective:**
           $$\\mathcal{L} = 0.5 \\cdot \\mathcal{L}_{\\text{Dice}} + 0.5 \\cdot \\mathcal{L}_{\\text{Focal}}$$
        """)

    with c_dgx:
        st.markdown("### ⚡ NVIDIA DGX B200 Hardware Specs")
        st.markdown("""
        | Parameter | Specification |
        |---|---|
        | **Hardware Platform** | NVIDIA DGX B200 (Blackwell Architecture) |
        | **GPU Instance** | NVIDIA B200 MIG Slice (Compute 10.0 / `sm_100`) |
        | **GPU VRAM** | **23 GB HBM3e High-Bandwidth Memory** |
        | **System RAM** | 32 GB RAM |
        | **Precision** | Native BFloat16 (BF16) AMP |
        | **Patch Size** | $128 \\times 128 \\times 128$ isotropic voxels |
        | **Batch Size** | 4 full 3D patches per step |
        | **Software Stack** | PyTorch 2.x, CUDA 12.8, SimpleITK, NiBabel |
        | **Training Duration** | Overnight (~13 hours, 4 models evaluated) |
        """)

    st.markdown("---")
    st.markdown("### 👥 Project Seminar Information")
    p1, p2, p3 = st.columns(3)
    p1.markdown("""
    **Project Title:**  
    Brain Tumor Segmentation in MRI Using Attention-Based 3D CNNs
    """)
    p2.markdown("""
    **Presenters:**  
    - Rishi Agrawal (C1-17)  
    - Rishil Pawar (C1-18)  
    - Sahil Deotale (C2-24)  
    - Shrishti Lal (C3-39)  
    """)
    p3.markdown("""
    **Supervised By:**  
    Dr. Uma Yadav  
    Department of Data Science  
    Shri Ramdeobaba College of Engineering & Management (RCOEM), Nagpur  
    """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("🧠 BraTS 2023 GLI Challenge · NVIDIA B200 Accelerated · RCOEM Nagpur CSE-DS 2026–2027")
