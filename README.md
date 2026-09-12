# 🧠 Brain Tumor Segmentation in MRI Using Attention-Based 3D CNNs

### RCOEM Nagpur — B.Tech CSE (Data Science) | Session 2026–2027 | Semester VII
**Team:** Rishi Agrawal (17) · Rishil Pawar (18) · Sahil Deotale (24) · Shrishti Lal (39)  
**Project Guide:** Dr. Uma Yadav, Assistant Professor

---

## 📁 Consolidated Project Structure

```
Brain-Tumor-Segmentation/
├── src/                                  # Modular Python source library
│   ├── __init__.py                       # Package exports
│   ├── config.py                         # Centralized config: paths, hyperparams, labels
│   ├── dataset.py                        # Subject-level zero-leakage splits, lazy loading, augmentation
│   ├── models.py                         # Unified 3D architecture supporting all 4 ablation variants
│   ├── losses.py                         # Soft Dice + Weighted Cross-Entropy + Focal loss
│   ├── metrics.py                        # 15-metric suite (WT, TC, ET) + robust HD95 computation
│   └── utils.py                          # Resumable checkpointing, fixed 100% coverage sliding-window
│
├── train.py                              # Unified CLI training script (--model_variant, --resume, etc.)
├── evaluate.py                           # Standalone CLI evaluation & Wilcoxon statistical testing script
├── demo_app.py                           # Interactive Streamlit web GUI
├── dgx_estimator.py                      # Hardware & resource estimator
├── notebooks/                            # 7-Step exploratory Jupyter notebooks
│   ├── 01_preprocess.ipynb
│   ├── 02_explore_data.ipynb
│   ├── 03_train_baseline.ipynb
│   ├── 04_train_attention_unet.ipynb
│   ├── 05_evaluate_compare.ipynb
│   ├── 06_ablation_study.ipynb
│   └── 07_visualise_results.ipynb
│
├── requirements.txt                      # Project dependencies
└── README.md
```

---

## 🚀 Quick Start & Training (DGX / Shared GPU Cluster)

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Unified CLI Training (`train.py`)
Run the unified training script directly. It supports background execution via `nohup` on shared GPU clusters with automatic crash/slot resumption.

```bash
# Train Proposed Dual-Attention 3D U-Net (Hybrid SE + Spatial Attention):
python train.py --model_variant hybrid --dataset_path "<UPDATE THIS TO YOUR ACTUAL DATASET LOCATION ON THE SERVER — e.g. /GPFS_CLUSTER/students/student02/Dataset_final>"

# Train Baseline 3D U-Net (for comparison):
python train.py --model_variant baseline --dataset_path "<UPDATE THIS TO YOUR ACTUAL DATASET LOCATION ON THE SERVER — e.g. /GPFS_CLUSTER/students/student02/Dataset_final>"

# Train Ablation Variants:
python train.py --model_variant channel   # Squeeze-and-Excitation channel attention only
python train.py --model_variant spatial   # Spatial Attention Gates only

# Resume an interrupted session cleanly from the latest checkpoint:
python train.py --model_variant hybrid --resume

# Run a 5-minute pipeline sanity check on 10 patients:
python train.py --model_variant hybrid --quick_test
```

### 3. Running in the Background via `nohup`
```bash
nohup python train.py --model_variant hybrid --dataset_path "<UPDATE THIS TO YOUR ACTUAL DATASET LOCATION ON THE SERVER — e.g. /GPFS_CLUSTER/students/student02/Dataset_final>" > train_hybrid.log 2>&1 &
```

### 4. Standalone Evaluation (`evaluate.py`)
Evaluate trained models on the held-out test split (15 clinical metrics computed across WT, TC, ET) and run paired Wilcoxon signed-rank statistical significance tests:

```bash
# Evaluate model checkpoint on the held-out test split:
python evaluate.py --checkpoint checkpoints/hybrid_unet3d_fold0_best.pth --model_variant hybrid

# Compare against baseline with paired Wilcoxon signed-rank test:
python evaluate.py --checkpoint checkpoints/hybrid_unet3d_fold0_best.pth --model_variant hybrid \
                   --compare_baseline checkpoints/baseline_unet3d_fold0_best.pth
```

### 5. Interactive Demo App (`demo_app.py`)
Launch the Streamlit web application for interactive 3D axial slice viewing and color-coded tumor overlays:
```bash
streamlit run demo_app.py
```

---

## 🔬 Core Methodological Rigor & Data Leakage Protections

1. **Patient-Level Splitting**: Scans are grouped strictly by Subject ID (`BraTS-GLI-XXXXX`). All temporal/multi-session scans of a patient remain in the same split (zero train/val/test leakage).
2. **Per-Case Intensity Normalization**: Z-score intensity normalization is computed strictly per-volume within the non-zero brain mask; no global dataset statistics leak across splits.
3. **Deterministic Evaluation**: Training patch extraction uses tumor-centered sampling (`foreground_prob=0.67`), whereas validation and testing use **mask-free 100% spatial coverage sliding-window inference with Gaussian blending**.
4. **Resumable State Restoration**: Checkpoint saving preserves model weights, optimizer state, LR scheduler state, AMP scaler state, epoch number, best metric, and RNG seed states.

---

## 📊 Comprehensive 15-Metric Suite

| Metric | Whole Tumor (WT) | Tumor Core (TC) | Enhancing Tumor (ET) |
| :--- | :---: | :---: | :---: |
| **Dice Similarity Score (DSC) ↑** | Evaluated | Evaluated | Evaluated |
| **Jaccard Index (IoU) ↑** | Evaluated | Evaluated | Evaluated |
| **95% Hausdorff Distance (HD95 mm) ↓** | Evaluated | Evaluated | Evaluated |
| **Sensitivity (Recall) ↑** | Evaluated | Evaluated | Evaluated |
| **Specificity ↑** | Evaluated | Evaluated | Evaluated |

---

## 🛠️ Tech Stack & Hardware Compatibility
- **Software**: Python 3.9–3.11 · PyTorch 2.x · NiBabel · SciPy · Scikit-Learn · MONAI · Streamlit · TensorBoard
- **Hardware Profile**: Optimized for 23GB VRAM MIG Slice / NVIDIA Blackwell B200 / 8-core CPU / 32GB RAM
