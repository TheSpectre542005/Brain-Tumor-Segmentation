#!/bin/bash
# =============================================================================
#  dgx_setup.sh — Run this once after SSH-ing into the DGX
#  RCOEM Brain Tumor Segmentation | BraTS 2023 GLI | Oct 2026
#
#  Usage:
#    chmod +x dgx_setup.sh
#    ./dgx_setup.sh
#
#  What it does:
#    1. Checks GPU / CUDA
#    2. Installs all Python dependencies
#    3. Unzips BraTS dataset (if zip is present)
#    4. Updates DATASET_PATH in src/config.py automatically
#    5. Runs dgx_healthcheck.py
#    6. Starts TensorBoard in background
# =============================================================================

set -e  # Exit immediately on any error

BOLD="\033[1m"
GREEN="\033[92m"
YELLOW="\033[93m"
RED="\033[91m"
RESET="\033[0m"

echo -e "${BOLD}============================================================${RESET}"
echo -e "${BOLD}  DGX Setup — Brain Tumor Segmentation (BraTS 2023 GLI)${RESET}"
echo -e "${BOLD}  RCOEM B.Tech CSE-DS 2026-27${RESET}"
echo -e "${BOLD}============================================================${RESET}"

# ── Step 1: GPU check ─────────────────────────────────────────────────────────
echo -e "\n${BOLD}[1/6] GPU Status${RESET}"
nvidia-smi --query-gpu=name,memory.total,memory.free,temperature.gpu \
           --format=csv,noheader,nounits | \
awk -F',' '{printf "  GPU: %s | VRAM: %s MB total | %s MB free | Temp: %s°C\n", $1,$2,$3,$4}'
echo ""

# ── Step 2: Install dependencies ──────────────────────────────────────────────
echo -e "${BOLD}[2/6] Installing Python dependencies${RESET}"
pip install --quiet --upgrade pip

# Check if PyTorch with CUDA is already working
python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "  PyTorch (CUDA) already installed."
else
    echo "  Installing PyTorch (CUDA 12.4)..."
    pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu124
fi

# Install remaining requirements
echo "  Installing project requirements..."
pip install --quiet -r requirements.txt

echo -e "  ${GREEN}✅ Dependencies installed${RESET}"

# ── Step 3: Locate / extract dataset ─────────────────────────────────────────
echo -e "\n${BOLD}[3/6] Dataset Setup${RESET}"

# Search in user HOME first, then root as fallback
TRAIN_DIR=$(find "$HOME" -maxdepth 4 -name "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData" -type d 2>/dev/null | head -1)
if [ -z "$TRAIN_DIR" ]; then
    TRAIN_DIR=$(find / -name "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData" -type d 2>/dev/null | head -1)
fi

if [ -z "$TRAIN_DIR" ]; then
    echo "  Training data not found extracted. Looking for zip in $HOME..."
    TRAIN_ZIP=$(find "$HOME" -name "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip" -type f 2>/dev/null | head -1)

    if [ -z "$TRAIN_ZIP" ]; then
        echo -e "  ${RED}❌ Neither extracted folder nor zip found!${RESET}"
        TRAIN_DIR="$HOME/brats2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    else
        echo "  Found zip: $TRAIN_ZIP"
        EXTRACT_DIR=$(dirname "$TRAIN_ZIP")
        echo "  Extracting (this takes ~5 minutes)..."
        unzip -q "$TRAIN_ZIP" -d "$EXTRACT_DIR"
        TRAIN_DIR="$EXTRACT_DIR/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
        echo -e "  ${GREEN}✅ Extracted to: $TRAIN_DIR${RESET}"
    fi
else
    echo -e "  ${GREEN}✅ Training data found: $TRAIN_DIR${RESET}"
fi

# Find validation data (search $HOME)
VAL_DIR=$(find "$HOME" -maxdepth 4 -name "ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData" -type d 2>/dev/null | head -1)
if [ -z "$VAL_DIR" ]; then
    VAL_ZIP=$(find "$HOME" -name "ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData.zip" -type f 2>/dev/null | head -1)
    if [ -n "$VAL_ZIP" ]; then
        EXTRACT_DIR=$(dirname "$VAL_ZIP")
        echo "  Extracting validation zip..."
        unzip -q "$VAL_ZIP" -d "$EXTRACT_DIR"
        VAL_DIR="$EXTRACT_DIR/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData"
    else
        VAL_DIR="$HOME/brats2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData"
    fi
fi

# Directory for preprocessed data inside user's directory (avoids permission errors)
PREPROC_DIR="$HOME/brats2023/BraTS2023_Preprocessed"
mkdir -p "$PREPROC_DIR"

# Count patients
N_PATIENTS=$(find "$TRAIN_DIR" -maxdepth 1 -name "BraTS-GLI-*" -type d 2>/dev/null | wc -l)
echo "  Patients found: $N_PATIENTS"

# ── Step 4: Update config.py ──────────────────────────────────────────────────
echo -e "\n${BOLD}[4/6] Updating src/config.py with DGX paths${RESET}"

# Escape paths for sed
TRAIN_ESC=$(echo "$TRAIN_DIR" | sed 's/[\/&]/\\&/g')
VAL_ESC=$(echo "$VAL_DIR"     | sed 's/[\/&]/\\&/g')
PREPROC_ESC=$(echo "$PREPROC_DIR" | sed 's/[\/&]/\\&/g')

sed -i "s|DATASET_PATH.*=.*|DATASET_PATH      = \"$TRAIN_DIR\"|" src/config.py
sed -i "s|VALIDATION_PATH.*=.*|VALIDATION_PATH   = \"$VAL_DIR\"|" src/config.py
sed -i "s|PREPROCESSED_PATH.*=.*|PREPROCESSED_PATH = \"$PREPROC_DIR\"|" src/config.py

# Also auto-patch dataset path in all notebooks so user doesn't have to edit them
sed -i "s|/home/yourname/BraTS2023_Training_Data|$TRAIN_DIR|g" notebooks/*.ipynb 2>/dev/null || true

echo "  DATASET_PATH      = $TRAIN_DIR"
echo "  VALIDATION_PATH   = $VAL_DIR"
echo "  PREPROCESSED_PATH = $PREPROC_DIR"
echo -e "  ${GREEN}✅ config.py and notebooks updated${RESET}"

# ── Step 5: Run health check ──────────────────────────────────────────────────
echo -e "\n${BOLD}[5/6] Running Pipeline Health Check${RESET}"
echo "  (This takes ~5 minutes — checks GPU, data, model, loss, VRAM...)"
echo ""

python dgx_healthcheck.py
HEALTH_EXIT=$?

if [ $HEALTH_EXIT -ne 0 ]; then
    echo -e "\n${RED}⛔ Health check FAILED. Fix the issues above before training.${RESET}"
    exit 1
fi

# ── Step 6: Start TensorBoard ─────────────────────────────────────────────────
echo -e "\n${BOLD}[6/6] Starting TensorBoard${RESET}"
mkdir -p ./logs
nohup tensorboard --logdir=./logs --port=6006 --host=0.0.0.0 \
      > ./logs/tensorboard.log 2>&1 &
TB_PID=$!
echo "  TensorBoard PID: $TB_PID"
echo "  Access at: http://<DGX_IP>:6006"
echo "  Log: ./logs/tensorboard.log"
echo -e "  ${GREEN}✅ TensorBoard started${RESET}"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}============================================================${RESET}"
echo -e "${GREEN}${BOLD}  ✅ DGX SETUP COMPLETE — Ready to Train!${RESET}"
echo -e "${BOLD}============================================================${RESET}"
echo ""
echo -e "  ${BOLD}Next steps:${RESET}"
echo "  1. jupyter lab --no-browser --port=8888 --ip=0.0.0.0"
echo "  2. Open in your browser: http://<DGX_IP>:8888"
echo "  3. Open notebooks/03_train_baseline.ipynb"
echo "  4. Confirm QUICK_TEST=True, run all cells, check Dice > 0.30 at epoch 5"
echo "  5. Set QUICK_TEST=False in src/config.py"
echo "  6. Rerun notebook for FULL training (~12h baseline, ~15h attention)"
echo ""
echo "  Monitor GPU while training:"
echo "    watch -n 5 nvidia-smi"
echo ""
