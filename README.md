# Brain Tumor Segmentation in MRI Using Attention-Based CNNs
### RCOEM Nagpur — B.Tech CSE (Data Science) | Session 2026–2027 | Semester VII
**Team:** Rishi Agrawal (17) · Rishil Pawar (18) · Sahil Deotale (24) · Shrishti Lal (39)  
**Guide:** Dr. Uma Yadav, Assistant Professor

---

## 📁 Project Structure

```
Brain_Tumor_Segmentation/
├── notebooks/
│   ├── 01_preprocess.ipynb          ← Skull strip, normalise, save .npz files
│   ├── 02_explore_data.ipynb        ← Visualise MRI slices + class balance
│   ├── 03_train_baseline.ipynb      ← Train plain 3D U-Net (baseline)
│   ├── 04_train_attention_unet.ipynb ← Train proposed Attention U-Net
│   ├── 05_evaluate_compare.ipynb    ← All metrics + comparison table
│   ├── 06_ablation_study.ipynb      ← Channel vs Spatial vs Hybrid attention
│   └── 07_visualise_results.ipynb   ← Attention maps + overlays + Grad-CAM
│
├── src/
│   ├── config.py                    ← Central configuration (change dataset path here)
│   ├── dataset.py                   ← BraTS 2023 GLI DataLoader + augmentation
│   ├── losses.py                    ← DiceLoss + FocalLoss + CombinedLoss
│   ├── metrics.py                   ← Dice, IoU, HD95, Sensitivity, Specificity
│   ├── utils.py                     ← Training loops, checkpointing, sliding window
│   └── models/
│       ├── unet3d.py                ← Baseline 3D U-Net
│       └── attention_unet3d.py      ← Proposed: SE + Spatial Attention Gates
│
├── checkpoints/                     ← Model checkpoints (auto-created)
├── results/                         ← CSV results + figures (auto-created)
├── logs/                            ← TensorBoard logs (auto-created)
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start (on DGX)

### Step 1 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Extract Dataset
```bash
cd /home/yourname
unzip ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip -d BraTS2023_Training_Data
unzip ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData.zip -d BraTS2023_Validation_Data
```

### Step 3 — Change ONE Line in src/config.py
```python
DATASET_PATH = "/home/yourname/BraTS2023_Training_Data"  # ← your actual path
```

### Step 4 — Start JupyterLab
```bash
jupyter lab --no-browser --port=8888 --ip=0.0.0.0
# Open in laptop browser: http://dgx_ip_address:8888
```

### Step 5 — Run Notebooks in Order
```
Day 1:  01_preprocess.ipynb       (~1.5 hrs)
Day 2:  03_train_baseline.ipynb   (~8–10 hrs, auto-checkpoints)
Day 3:  04_train_attention_unet   (~12–14 hrs, auto-checkpoints)
Day 4:  05_evaluate_compare       (~30 min)
Day 5:  06_ablation_study         (~2 hrs)
Day 6:  07_visualise_results      (~30 min) — demo ready!
```

---

## ⚡ Quick Test Mode
Before committing to full training, test the pipeline on 10 patients:
```python
QUICK_TEST = True  # Set in the config cell of each notebook
```
This runs in ~5 minutes and verifies the entire pipeline works.

---

## 📊 Expected Results

| Model | Dice WT | Dice TC | Dice ET | HD95 WT↓ | HD95 ET↓ |
|-------|---------|---------|---------|----------|----------|
| 3D U-Net (Baseline) | ~0.84 | ~0.79 | ~0.74 | ~6.5 mm | ~9.5 mm |
| Channel Attn only   | ~0.86 | ~0.81 | ~0.76 | ~6.1 mm | ~8.2 mm |
| Spatial Attn only   | ~0.86 | ~0.82 | ~0.77 | ~5.8 mm | ~7.5 mm |
| **Attn U-Net (Ours)** | **~0.87** | **~0.83** | **~0.78** | **~5.1 mm** | **~6.8 mm** |

---

## 🧠 Architecture

### Baseline (3D U-Net)
Standard encoder-decoder with MaxPool downsampling and TransposedConv upsampling.

### Proposed (Attention U-Net 3D)
Adds two complementary attention mechanisms:
1. **Squeeze-and-Excitation (SE) blocks** — Channel attention at every encoder/decoder level
2. **Spatial Attention Gates** — Focus decoder on relevant tumor regions, suppress background

---

## 📈 Metrics Computed

| Metric | What it measures |
|--------|-----------------|
| **Dice Score** | Overlap between prediction and ground truth (↑ better) |
| **IoU** | Intersection-over-union (↑ better) |
| **HD95** | 95th percentile surface distance in mm (↓ better) |
| **Sensitivity** | True positive rate — "finds all tumor" (↑ better) |
| **Specificity** | True negative rate — "avoids false alarms" (↑ better) |

All metrics computed for **3 sub-regions**: WT (Whole Tumor), TC (Tumor Core), ET (Enhancing Tumor)  
→ **15 numbers per model** reported in the final table.

---

## 💾 Auto-Checkpointing
All training notebooks auto-save every epoch. If your DGX slot ends mid-training:
```python
RESUME = True  # (default) — automatically resumes from latest checkpoint
```

---

## 🛠️ Tech Stack
Python 3 · PyTorch · MONAI · NiBabel · SimpleITK · NumPy · Scikit-learn · Matplotlib · TensorBoard
