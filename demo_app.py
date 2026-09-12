"""
demo_app.py — Brain Tumor Segmentation Demo
============================================
Run with:   streamlit run demo_app.py
Install:    pip install streamlit nibabel matplotlib torch

Upload a BraTS .nii.gz file → see the predicted segmentation overlay.
Works with:
  • Pre-trained checkpoint (after DGX training)
  • Ground-truth mode for presentation before training is done
"""

import io
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor Segmentation Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background-color: #0d1117; }

    .hero {
        background: linear-gradient(135deg, #1a1f2e 0%, #0f3460 50%, #16213e 100%);
        border-radius: 16px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid #30363d;
    }
    .hero h1 { color: #58a6ff; font-size: 2.2rem; font-weight: 700; margin: 0; }
    .hero p  { color: #8b949e; font-size: 1.05rem; margin-top: 0.5rem; }

    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-val { font-size: 2rem; font-weight: 700; color: #3fb950; }
    .metric-lbl { font-size: 0.85rem; color: #8b949e; margin-top: 0.2rem; }

    .region-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    .stButton>button {
        background: linear-gradient(90deg, #1f6feb, #388bfd);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        width: 100%;
    }
    .stButton>button:hover { opacity: 0.9; }

    .info-box {
        background: #161b22;
        border-left: 4px solid #58a6ff;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        color: #c9d1d9;
    }
    div[data-testid="stSidebar"] { background-color: #161b22; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# COLOUR MAP
# =============================================================================
SEG_COLORS = {
    0: (0,   0,   0,   0  ),   # Background — transparent
    1: (255, 100, 100, 180),   # NCR (Necrotic Core) — red
    2: (100, 180, 255, 180),   # ED  (Edema)         — blue
    3: (100, 255, 150, 200),   # ET  (Enhancing)     — green
}

LABEL_NAMES = {
    0: "Background",
    1: "Necrotic Core (NCR)",
    2: "Peritumoral Edema (ED)",
    3: "Enhancing Tumor (ET)",
}

REGION_COLORS = {
    "NCR": "#ff6464",
    "ED":  "#64b4ff",
    "ET":  "#64ff96",
}


def seg_to_rgba(seg_slice: np.ndarray) -> np.ndarray:
    """Convert 2D segmentation label map → RGBA overlay."""
    h, w = seg_slice.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for label, color in SEG_COLORS.items():
        mask = seg_slice == label
        rgba[mask] = color
    return rgba


def normalize_slice(s: np.ndarray) -> np.ndarray:
    """Normalise a 2D slice to [0, 1] for display."""
    mn, mx = s.min(), s.max()
    if mx == mn:
        return np.zeros_like(s, dtype=np.float32)
    return ((s - mn) / (mx - mn)).astype(np.float32)


def make_overlay_figure(mri_slice, seg_slice, title="", alpha=0.45):
    """Return a matplotlib figure with MRI + segmentation overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5),
                             facecolor="#0d1117", gridspec_kw={"wspace": 0.05})

    mri_norm = normalize_slice(mri_slice)
    rgba     = seg_to_rgba(seg_slice)

    panels = [
        (mri_norm,                      "MRI (FLAIR)",       "gray"),
        (np.zeros_like(mri_norm),       "Segmentation",      "gray"),
        (mri_norm,                      "Overlay",           "gray"),
    ]

    for ax, (img, lbl, cmap) in zip(axes, panels):
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        if lbl == "Segmentation":
            ax.imshow(rgba)
        if lbl == "Overlay":
            ax.imshow(rgba, alpha=alpha)
        ax.set_title(lbl, color="#c9d1d9", fontsize=11, pad=6)
        ax.axis("off")
        ax.set_facecolor("#0d1117")

    if title:
        fig.suptitle(title, color="#58a6ff", fontsize=13, y=1.01)

    # Legend
    patches = [
        mpatches.Patch(color=np.array(SEG_COLORS[1][:3])/255, label="NCR — Necrotic Core"),
        mpatches.Patch(color=np.array(SEG_COLORS[2][:3])/255, label="ED  — Edema"),
        mpatches.Patch(color=np.array(SEG_COLORS[3][:3])/255, label="ET  — Enhancing Tumor"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               facecolor="#161b22", edgecolor="#30363d",
               labelcolor="#c9d1d9", fontsize=9, framealpha=1,
               bbox_to_anchor=(0.5, -0.08))
    return fig


def compute_dice(pred, gt, label):
    p = (pred == label).astype(np.float32)
    g = (gt   == label).astype(np.float32)
    intersection = (p * g).sum()
    denom = p.sum() + g.sum()
    return (2 * intersection / denom) if denom > 0 else 1.0


def compute_region_dice(pred, gt):
    """BraTS sub-region Dice."""
    wt_p = (pred  > 0).astype(np.float32); wt_g = (gt  > 0).astype(np.float32)
    tc_p = np.isin(pred, [1,3]).astype(np.float32); tc_g = np.isin(gt, [1,3]).astype(np.float32)
    et_p = (pred == 3).astype(np.float32); et_g = (gt == 3).astype(np.float32)
    def _dice(p, g):
        inter = (p*g).sum(); d = p.sum()+g.sum()
        return float(2*inter/d) if d > 0 else 1.0
    return {"WT": _dice(wt_p, wt_g), "TC": _dice(tc_p, tc_g), "ET": _dice(et_p, et_g)}


# =============================================================================
# MOCK INFERENCE  (used when no checkpoint is available)
# =============================================================================
def mock_predict(mri_vol: np.ndarray) -> np.ndarray:
    """
    Simulate a segmentation prediction from MRI volume.
    Uses a simple intensity-threshold heuristic — purely for demo purposes.
    Replace with real model inference once training is done.
    """
    # Use FLAIR channel (index 0 if 4-ch, else the only channel)
    flair = mri_vol[..., 0] if mri_vol.ndim == 4 else mri_vol
    flair_norm = (flair - flair.min()) / (flair.max() - flair.min() + 1e-8)

    seg = np.zeros_like(flair_norm, dtype=np.int32)
    seg[flair_norm > 0.70] = 3   # ET  — brightest
    seg[flair_norm > 0.50] = 1   # NCR — bright
    seg[flair_norm > 0.35] = 2   # ED  — moderately bright
    seg[flair_norm > 0.70] = 3   # ET  — restore ET on top

    # Keep only a blob near the centroid (crude tumour localisation)
    from scipy.ndimage import label as ndlabel, binary_dilation
    binary = seg > 0
    labeled, _ = ndlabel(binary)
    if labeled.max() > 0:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        largest = sizes.argmax()
        seg[labeled != largest] = 0

    return seg


# =============================================================================
# REAL MODEL INFERENCE  (loads checkpoint)
# =============================================================================
@st.cache_resource
def load_model(ckpt_path: str):
    if not TORCH_OK:
        return None
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from src.models.attention_unet3d import AttentionUNet3D
        model = AttentionUNet3D(in_channels=4, out_channels=4, init_features=32)
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        model.eval()
        return model
    except Exception as e:
        st.warning(f"Could not load model: {e}")
        return None


def real_predict(model, mri_vol: np.ndarray) -> np.ndarray:
    """Run sliding-window inference with the trained model."""
    try:
        from monai.inferers import sliding_window_inference
        x = torch.tensor(mri_vol).float()
        if x.ndim == 3:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 4:
            x = x.permute(3, 0, 1, 2).unsqueeze(0)
        with torch.no_grad():
            logits = sliding_window_inference(
                x, roi_size=(96, 96, 96), sw_batch_size=1,
                predictor=model, overlap=0.5,
            )
        pred = logits.argmax(dim=1).squeeze(0).numpy().astype(np.int32)
        return pred
    except Exception as e:
        st.warning(f"Inference error: {e} — falling back to demo mode.")
        return mock_predict(mri_vol)


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    ckpt_path = st.text_input(
        "Checkpoint path (optional)",
        placeholder="/home/agrawalrb_1/checkpoints/attention_unet_best.pth",
        help="Leave blank to run in demo mode (no trained model needed)",
    )
    use_gt_as_pred = st.checkbox(
        "Use ground truth as prediction",
        value=True,
        help="Tick this for presentation before training is done — shows what the output WILL look like",
    )
    alpha = st.slider("Overlay opacity", 0.1, 0.9, 0.45, 0.05)

    st.markdown("---")
    st.markdown("### 📖 How to use")
    st.markdown("""
1. Upload **FLAIR** `.nii.gz` file
2. Upload **segmentation** `.nii.gz` file  
   *(ground truth from BraTS dataset)*
3. Adjust the slice slider
4. See the colour-coded tumor regions

**Tumor regions:**
- 🔴 **NCR** — Necrotic Core
- 🔵 **ED**  — Edema
- 🟢 **ET**  — Enhancing Tumor
""")

    st.markdown("---")
    st.markdown("**RCOEM 2026–27**")
    st.caption("Brain Tumor Segmentation | CSE-DS Sem VII")


# =============================================================================
# MAIN PAGE
# =============================================================================
st.markdown("""
<div class="hero">
  <h1>🧠 Brain Tumor Segmentation</h1>
  <p>Attention-Based 3D CNN · BraTS 2023 · RCOEM Nagpur</p>
  <p style="font-size:0.85rem; color:#58a6ff;">
    Rishi Agrawal · Rishil Pawar · Sahil Deotale · Shrishti Lal &nbsp;|&nbsp; Guide: Dr. Uma Yadav
  </p>
</div>
""", unsafe_allow_html=True)

# ── File Uploaders ────────────────────────────────────────────────────────────
col_u1, col_u2 = st.columns(2)
with col_u1:
    flair_file = st.file_uploader(
        "📁 Upload FLAIR MRI  (.nii or .nii.gz)",
        type=["nii", "gz"],
        key="flair",
    )
with col_u2:
    seg_file = st.file_uploader(
        "📁 Upload Segmentation mask  (.nii or .nii.gz)",
        type=["nii", "gz"],
        key="seg",
        help="Ground truth from BraTS dataset (e.g. BraTS-GLI-00000-000-seg.nii.gz)",
    )

if not NIBABEL_OK:
    st.error("❌ nibabel is not installed.  Run:  `pip install nibabel`")
    st.stop()

if not flair_file or not seg_file:
    st.markdown("""
    <div class="info-box">
    <b>👆 Upload a FLAIR and segmentation file above to begin.</b><br><br>
    Files are in your BraTS dataset folder, e.g.:<br>
    <code>BraTS-GLI-00000-000-t2f.nii.gz</code>  ← FLAIR<br>
    <code>BraTS-GLI-00000-000-seg.nii.gz</code>  ← Segmentation
    </div>
    """, unsafe_allow_html=True)

    # Show a sample layout with placeholder info
    st.markdown("### 📊 What the output looks like")
    st.markdown("""
    | Panel | Description |
    |-------|-------------|
    | **MRI (FLAIR)** | Raw MRI scan slice — grayscale |
    | **Segmentation** | Colour-coded tumor regions only |
    | **Overlay** | Both combined — tumor highlighted on MRI |
    """)
    st.stop()


# ── Load volumes ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading MRI volume...")
def load_nifti(file_bytes, name):
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
        f.write(file_bytes)
        tmp = f.name
    img = nib.load(tmp)
    return img.get_fdata().astype(np.float32)


flair_vol = load_nifti(flair_file.read(), flair_file.name)
seg_vol   = load_nifti(seg_file.read(),   seg_file.name)

# Squeeze to 3D if needed
if flair_vol.ndim == 4:
    flair_vol = flair_vol[..., 0]
seg_vol = seg_vol.squeeze()

n_slices = flair_vol.shape[2]

# ── Run inference ─────────────────────────────────────────────────────────────
if use_gt_as_pred:
    pred_vol = seg_vol.copy()
    mode_label = "🟡 Showing Ground Truth as Prediction (Demo Mode)"
elif ckpt_path and Path(ckpt_path).exists():
    model = load_model(ckpt_path)
    if model:
        with st.spinner("🔄 Running model inference..."):
            pred_vol = real_predict(model, flair_vol)
        mode_label = "🟢 Trained Model Prediction"
    else:
        pred_vol = mock_predict(flair_vol)
        mode_label = "🔴 Heuristic Demo Mode (no checkpoint)"
else:
    pred_vol = mock_predict(flair_vol)
    mode_label = "🔴 Heuristic Demo Mode (no checkpoint)"

st.info(mode_label)

# ── Dice Scores ───────────────────────────────────────────────────────────────
dice = compute_region_dice(pred_vol.astype(np.int32), seg_vol.astype(np.int32))

st.markdown("### 📈 Dice Scores")
c1, c2, c3, c4 = st.columns(4)
metrics = [
    ("Whole Tumor (WT)", dice["WT"], "All tumor regions combined"),
    ("Tumor Core (TC)",  dice["TC"], "NCR + Enhancing Tumor"),
    ("Enhancing Tumor (ET)", dice["ET"], "Most aggressive part"),
    ("Mean Dice",        np.mean(list(dice.values())), "Average across regions"),
]
for col, (lbl, val, tip) in zip([c1, c2, c3, c4], metrics):
    color = "#3fb950" if val >= 0.80 else ("#f0883e" if val >= 0.60 else "#f85149")
    col.markdown(f"""
    <div class="metric-card">
      <div class="metric-val" style="color:{color};">{val:.3f}</div>
      <div class="metric-lbl">{lbl}</div>
      <div style="font-size:0.75rem;color:#6e7681;margin-top:0.3rem;">{tip}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Slice viewer ──────────────────────────────────────────────────────────────
st.markdown("### 🔬 Slice Viewer")

# Find a good default slice (one with most tumor)
tumor_counts = [(seg_vol[:, :, i] > 0).sum() for i in range(n_slices)]
best_slice   = int(np.argmax(tumor_counts))

slice_idx = st.slider(
    "Axial slice", 0, n_slices - 1, best_slice,
    help="Slide to browse through brain slices. Tumor is most visible around the middle.",
)

flair_slice = flair_vol[:, :, slice_idx].T
seg_slice   = seg_vol[:, :, slice_idx].T.astype(np.int32)
pred_slice  = pred_vol[:, :, slice_idx].T.astype(np.int32)

fig = make_overlay_figure(
    flair_slice, pred_slice,
    title=f"Axial Slice {slice_idx} / {n_slices - 1}",
    alpha=alpha,
)
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# ── Per-slice tumor stats ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📊 Tumor Region Breakdown — This Slice")

labels_present = np.unique(pred_slice[pred_slice > 0])
if len(labels_present) == 0:
    st.caption("No tumor detected in this slice. Try a different slice.")
else:
    cols = st.columns(len(labels_present))
    for col, lbl in zip(cols, labels_present):
        px_count = (pred_slice == lbl).sum()
        name = LABEL_NAMES[lbl]
        col.metric(name, f"{px_count:,} px")

# ── Download ──────────────────────────────────────────────────────────────────
buf = io.BytesIO()
fig2 = make_overlay_figure(flair_slice, pred_slice, alpha=alpha)
fig2.savefig(buf, format="png", dpi=150, bbox_inches="tight",
             facecolor="#0d1117")
plt.close(fig2)
buf.seek(0)

st.download_button(
    "⬇️  Download this overlay as PNG",
    data=buf,
    file_name=f"segmentation_slice_{slice_idx}.png",
    mime="image/png",
)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Brain Tumor Segmentation · RCOEM B.Tech CSE-DS 2026–27 · "
    "Guide: Dr. Uma Yadav · Model: 3D Attention U-Net (SE + Spatial Gates)"
)
