# 🚀 Complete Guide: Running Brain Tumor Segmentation on NVIDIA DGX B200

> **Project:** Brain Tumor Segmentation in MRI Using Attention-Based 3D CNNs  
> **Team:** Rishi Agrawal (17) · Rishil Pawar (18) · Sahil Deotale (24) · Shrishti Lal (39)  
> **Institution:** RCOEM Nagpur — B.Tech CSE (Data Science) | Semester VII (2026–2027)  
> **Target Hardware:** NVIDIA Blackwell B200 GPU (DGX B200 / MIG Slice)  

---

## 📑 Table of Contents
1. [Overview & Blackwell B200 Hardware Advantages](#1-overview--blackwell-b200-hardware-advantages)
2. [Phase 0 — SSH Connection & Port Forwarding](#phase-0--ssh-connection--port-forwarding)
3. [Phase 1 — File & Dataset Transfer to DGX](#phase-1--file--dataset-transfer-to-dgx)
4. [Phase 2 — Environment & PyTorch CUDA Setup](#phase-2--environment--pytorch-cuda-setup)
5. [Phase 3 — Hardware Sanity & Pipeline Health Check](#phase-3--hardware-sanity--pipeline-health-check)
6. [Phase 4 — Configuring `src/config.py` for B200](#phase-4--configuring-srcconfigpy-for-b200)
7. [Phase 5 — Step-by-Step Training & Notebook Execution](#phase-5--step-by-step-training--notebook-execution)
8. [Phase 6 — Headless / Tmux Execution (Crash-Proof Training)](#phase-6--headless--tmux-execution-crash-proof-training)
9. [Phase 7 — Live Monitoring (GPU & TensorBoard)](#phase-7--live-monitoring-gpu--tensorboard)
10. [Troubleshooting & DGX Failure Recovery Guide](#10-troubleshooting--dgx-failure-recovery-guide)

---

## 1. Overview & Blackwell B200 Hardware Advantages

The **NVIDIA Blackwell B200** is NVIDIA's flagship AI accelerator (Compute Capability 10.0 / `sm_100`). In comparison to consumer laptop GPUs (e.g. RTX 4060 with 8 GB VRAM), the B200 provides:

| Spec / Feature | NVIDIA B200 MIG Slice | Full NVIDIA B200 | RTX 4060 Laptop |
|---|---|---|---|
| **VRAM** | 23 GB HBM3e | 180 GB HBM3e | 8 GB GDDR6 |
| **Memory Bandwidth** | ~1.1 TB/s | ~8.0 TB/s | ~272 GB/s |
| **Tensor Cores** | 5th Gen (FP4/FP8/BF16/TF32) | 5th Gen | 4th Gen |
| **Native Precision** | BF16 (No loss scaling needed) | BF16 / FP8 | FP16 (Risk of underflow) |
| **Batch Size (128³ Patch)** | Batch Size = 4 | Batch Size = 8–16 | Cannot fit 128³ (OOM) |
| **Total Training Time** | ~12–15 hrs (Full 100 epochs) | ~3–5 hrs | Days (or crashes) |

### Key B200 Optimizations Enabled in this Repo:
1. **BFloat16 (BF16) AMP:** Configured in `src/utils.py` and `src/config.py` (`USE_AMP = True`). BF16 retains the dynamic range of FP32 (8-bit exponent), eliminating gradient underflow without requiring `GradScaler`.
2. **TensorFloat-32 (TF32):** Automatically leverages 5th-Gen Blackwell Tensor Cores for single-precision matrix multiplications and 3D convolutions.
3. **Patch Resolution:** Scaled up to $128 \times 128 \times 128$ voxels (from the laptop limit of $96^3$), providing significantly larger spatial anatomical context for tumor boundaries.

---

## Phase 0 — SSH Connection & Port Forwarding

Because university firewalls block internal web ports, you should establish an SSH tunnel with **port forwarding** for both **JupyterLab (8888)** and **TensorBoard (6006)**.

### On your Laptop (PowerShell or Terminal):
```bash
# Syntax: ssh -L <LocalPort>:localhost:<RemotePort> user@<DGX_IP>
ssh -L 8888:localhost:8888 -L 6006:localhost:6006 username@<dgx-ip-address>
```

*(Replace `username` and `<dgx-ip-address>` with your credentials provided by the DGX lab administrator).*

---

## Phase 1 — File & Dataset Transfer to DGX

You need two things on the DGX:
1. The **Codebase** (`Brain_Tumor_Segmentation/`)
2. The **BraTS 2023 Datasets** (`ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip` & `ValidationData.zip`)

### Option A: Via Command Line (SCP / Rsync)
Run from your local laptop:
```bash
# Transfer the project folder (excluding heavy virtualenvs/git objects if desired)
scp -r "c:\Users\Rishil\Documents\Brain_Tumor_Segmentation" username@<dgx-ip>:~/Brain_Tumor_Segmentation

# Transfer the Training Dataset Zip (~13 GB)
scp "c:\Users\Rishil\Documents\Brain_Tumor_Segmentation\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip" username@<dgx-ip>:~/brats2023/

# Transfer the Validation Dataset Zip (~2.3 GB)
scp "c:\Users\Rishil\Documents\Brain_Tumor_Segmentation\ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData.zip" username@<dgx-ip>:~/brats2023/
```

### Option B: Via WinSCP (GUI)
1. Open **WinSCP** on Windows.
2. Protocol: **SFTP**, Host Name: `<dgx-ip>`, Port: `22`, enter your Username and Password.
3. Drag and drop the `Brain_Tumor_Segmentation` folder and the `.zip` files into `/home/username/`.

### Extracting Dataset on DGX:
SSH into the DGX and run:
```bash
mkdir -p ~/brats2023
cd ~/brats2023

# Move zips here if they were copied to home
mv ~/ASNR-MICCAI-BraTS2023*.zip ~/brats2023/ 2>/dev/null || true

# Extract Training Data (~13 GB, takes ~3-5 mins)
unzip -q ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip

# Extract Validation Data (~2.3 GB, takes ~1 min)
unzip -q ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData.zip

# Verify extracted folders
ls -lh ~/brats2023/
# Expected:
#   ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/
#   ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData/
```

---

## Phase 2 — Environment & PyTorch CUDA Setup

On the DGX, navigate to your project directory and set up a Python virtual environment:

```bash
cd ~/Brain_Tumor_Segmentation

# 1. Create and activate a clean virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Upgrade pip
pip install --upgrade pip

# 3. Install PyTorch with CUDA 12.4+ (required for Blackwell B200)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 4. Install all project dependencies
pip install -r requirements.txt
```

> **Automated Setup Shortcut:**  
> The repository includes an all-in-one setup script:
> ```bash
> chmod +x dgx_setup.sh
> ./dgx_setup.sh
> ```
> This script automatically verifies GPU availability, installs dependencies, unpacks datasets, updates `src/config.py`, runs the health check, and starts TensorBoard in the background.

---

## Phase 3 — Hardware Sanity & Pipeline Health Check

Before committing to a 10-hour training run, run the diagnostics:

### 1. Check GPU Visibility & VRAM
```bash
nvidia-smi
```
Verify that an **NVIDIA B200** or an **NVIDIA B200 MIG (e.g. 1g.24gb / ~23 GB)** is visible and recognized.

### 2. Run Health Check
```bash
python3 dgx_healthcheck.py
```
This runs a 12-point pipeline verification in ~3 minutes:
- [x] Python $\ge 3.9$ & PyTorch $\ge 2.0$
- [x] CUDA recognition & VRAM availability ($\ge 10\text{ GB}$)
- [x] Native **BF16 AMP** autocast functionality on Blackwell
- [x] Dataset folder discovery & NIfTI `.nii.gz` multi-channel loading
- [x] 3D patch extraction & foreground-biased sampling
- [x] Forward & backward pass for **Baseline 3D U-Net** and **Attention U-Net**
- [x] Combined Soft Dice + Focal loss computation
- [x] Checkpoint serialization (`.pth` save & restore)

If all items output `✅ OK`, your system is certified to train.

---

## Phase 4 — Configuring `src/config.py` for B200

Inspect `src/config.py` and verify settings matching your DGX allocation:

```python
# =============================================================================
# 1. Paths (Auto-configured by dgx_setup.sh or set manually)
# =============================================================================
HOME_DIR          = os.path.expanduser("~")
DATASET_PATH      = os.path.join(HOME_DIR, "brats2023", "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData")
VALIDATION_PATH   = os.path.join(HOME_DIR, "brats2023", "ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData")
PREPROCESSED_PATH = os.path.join(HOME_DIR, "brats2023", "BraTS2023_Preprocessed")

# =============================================================================
# 2. Hyperparameters for NVIDIA B200
# =============================================================================
# If on B200 MIG Slice (23 GB VRAM | 32 GB RAM):
PATCH_SIZE        = (128, 128, 128)  # 128³ patches provide high anatomical context
BATCH_SIZE        = 4                # 4 patches fit safely in 23 GB VRAM with BF16
NUM_WORKERS       = 3                # Keeps system RAM under 32 GB
USE_AMP           = True             # BF16 mixed precision on Blackwell

# If on FULL NVIDIA B200 (180 GB VRAM):
# BATCH_SIZE      = 8 or 16          # Can scale batch size for even faster epochs
# NUM_WORKERS     = 6 or 8

# =============================================================================
# 3. Quick Test vs Full Run
# =============================================================================
QUICK_TEST        = True             # Set to True FIRST to test 10 patients in ~5 min
# Set QUICK_TEST = False when ready for full 100-epoch training!
```

---

## Phase 5 — Step-by-Step Training & Notebook Execution

### Starting JupyterLab:
```bash
# Activate environment
source ~/Brain_Tumor_Segmentation/venv/bin/activate
cd ~/Brain_Tumor_Segmentation

# Launch JupyterLab (bound to localhost for SSH port forwarding)
jupyter lab --no-browser --port=8888 --ip=0.0.0.0
```

Now, on your **laptop's browser**, navigate to:
```
http://localhost:8888
```
*(Enter the security token printed in the DGX terminal).*

---

### Recommended Schedule of Execution

```
┌───────────┬───────────────────────────────────┬──────────────┬───────────────────────────────────────────┐
│ Step      │ Notebook / Script                 │ B200 Time    │ Key Output / Milestone                    │
├───────────┼───────────────────────────────────┼──────────────┼───────────────────────────────────────────┤
│ 1. Preproc│ notebooks/01_preprocess.ipynb     │ ~45 min      │ Normalized .npz patient volumes cached    │
│ 2. EDA    │ notebooks/02_explore_data.ipynb   │ ~15 min      │ Class balance & MRI slice inspection      │
│ 3. Smoke  │ notebooks/03_train_baseline.ipynb │ ~5 min       │ QUICK_TEST=True: verifies convergence     │
│ 4. Train1 │ notebooks/03_train_baseline.ipynb │ ~8–10 hrs    │ Baseline 3D U-Net checkpoint (.pth)       │
│ 5. Train2 │ notebooks/04_train_attention_unet │ ~12–14 hrs   │ Proposed Dual-Attention U-Net (.pth)      │
│ 6. Eval   │ notebooks/05_evaluate_compare.ipynb│ ~30 min     │ 15-Metric Comparison Table (WT, TC, ET)   │
│ 7. Ablate │ notebooks/06_ablation_study.ipynb │ ~2 hrs       │ Channel vs Spatial vs Combined Attention  │
│ 8. Visual │ notebooks/07_visualise_results.ipynb│ ~30 min    │ 3D Attention heatmaps & Grad-CAM overlays │
└───────────┴───────────────────────────────────┴──────────────┴───────────────────────────────────────────┘
```

---

## Phase 6 — Headless / Tmux Execution (Crash-Proof Training)

> [!WARNING]
> **Do not train long models directly in an open browser tab!**  
> If your laptop goes to sleep, loses Wi-Fi, or you close the browser, your JupyterLab web session can disconnect, killing the training cell.  
> Always use a detached **`tmux`** session.

### 1. Start a persistent `tmux` session:
```bash
# Create a session named 'train'
tmux new -s train
```

### 2. Run your training headless inside `tmux`:
You can execute any notebook directly via command line without opening a browser:

```bash
# Activate your environment
source ~/Brain_Tumor_Segmentation/venv/bin/activate
cd ~/Brain_Tumor_Segmentation

# Ensure QUICK_TEST = False in src/config.py for full training:
sed -i 's/QUICK_TEST        = True/QUICK_TEST        = False/' src/config.py

# Run Baseline 3D U-Net
jupyter nbconvert --to notebook --execute notebooks/03_train_baseline.ipynb \
    --output 03_train_baseline_executed.ipynb

# Once Baseline finishes, run Proposed Attention U-Net
jupyter nbconvert --to notebook --execute notebooks/04_train_attention_unet.ipynb \
    --output 04_train_attention_unet_executed.ipynb
```

### 3. Detach and close your laptop safely:
- Press `Ctrl + B`, then release and press `D`.
- You are now safely detached. You can close SSH, close your laptop, and walk away.
- When you reconnect later, re-attach to view live progress:
  ```bash
  tmux attach -t train
  ```

---

## Phase 7 — Live Monitoring (GPU & TensorBoard)

### 1. Monitor GPU Utilization & VRAM:
Open a second terminal or tmux window:
```bash
watch -n 2 nvidia-smi
```
- Look for **Volatile GPU-Util** $\approx 85\% - 98\%$.
- Look for **Memory-Usage** $\approx 18 - 21\text{ GB}$ (stable, no memory leaks).

### 2. Monitor Training Curves in TensorBoard:
On the DGX:
```bash
cd ~/Brain_Tumor_Segmentation
tensorboard --logdir=./logs --port=6006 --host=0.0.0.0
```

On your **laptop's browser**, navigate to:
```
http://localhost:6006
```
You can view real-time loss graphs, Soft Dice improvements, and learning rate decays as each epoch completes.

---

## 10. Troubleshooting & DGX Failure Recovery Guide

| Symptom / Error | Root Cause | Solution |
|---|---|---|
| `torch.cuda.OutOfMemoryError` | Batch size or patch size too large for allocated VRAM | In `src/config.py`: Reduce `BATCH_SIZE = 2` or set `PATCH_SIZE = (96, 96, 96)`. Ensure `USE_AMP = True`. |
| `DataLoader worker (pid X) is killed by signal: Bus error` | System `/dev/shm` (shared memory) exhausted | In `src/config.py`: Reduce `NUM_WORKERS = 2`. In DataLoader call, ensure `pin_memory = False` if host RAM is constrained. |
| `RuntimeError: CUDA error: no kernel image is available` | PyTorch was installed without Blackwell sm_100 support | Reinstall PyTorch with CUDA 12.4+ wheel: `pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124` |
| DGX Lab Slot ends mid-training | Session time limit reached | Checkpoints auto-save every epoch to `checkpoints/`. When your next slot starts, set `RESUME = True` in the notebook and re-run; training resumes seamlessly from the exact last saved epoch. |
| JupyterLab cell stops when laptop lid is closed | Network disconnect terminated the browser websocket | Use **Phase 6 (Tmux + `nbconvert`)** so execution runs entirely in a detached background daemon. |
| `No space left on device` | System `/tmp` or root drive is full | BraTS preprocessed data should be saved inside `~/brats2023/` (user home directory), not in `/tmp`. `dgx_setup.sh` sets `PREPROCESSED_PATH` inside user home by default. |

---

### ✅ Quick Reference Checklist for Lab Days:
- [ ] SSH with `-L 8888:localhost:8888 -L 6006:localhost:6006`
- [ ] Run in `tmux new -s train`
- [ ] Run `python3 dgx_healthcheck.py` $\to$ Confirm all 12 checks pass
- [ ] First run: `QUICK_TEST = True` $\to$ Confirm 5-min pipeline run
- [ ] Full run: `QUICK_TEST = False` $\to$ Launch training
- [ ] Check TensorBoard at `http://localhost:6006`
- [ ] Download best model checkpoints (`checkpoints/*.pth`) and results (`results/*.csv`) to your laptop when complete!
