"""
config.py — Central configuration for BraTS 2023 GLI Brain Tumor Segmentation
All notebooks import from here. Only change DATASET_PATH to match your DGX setup.

DGX B200 MIG Configuration (23 GB VRAM | 32 GB RAM | 8 CPU cores)
  Updated: October 2026 — optimised for Oct 17 submission deadline
"""

import os
from pathlib import Path

# =============================================================================
# SET THESE PATHS ON DGX BEFORE RUNNING ANYTHING
# =============================================================================
HOME_DIR = os.path.expanduser("~")
DATASET_PATH      = os.path.join(HOME_DIR, "Brain_Tumor_Segmentation", "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData")
VALIDATION_PATH   = os.path.join(HOME_DIR, "Brain_Tumor_Segmentation", "ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData")
PREPROCESSED_PATH = os.path.join(HOME_DIR, "brats2023", "BraTS2023_Preprocessed")

# Output paths (auto-created)
CHECKPOINT_DIR = "./checkpoints"
RESULTS_DIR    = "./results"
LOGS_DIR       = "./logs"

# =============================================================================
# BraTS 2023 GLI File Naming Convention
# =============================================================================
MODALITIES = {
    "flair": "t2f",
    "t1":    "t1n",
    "t1ce":  "t1c",
    "t2":    "t2w",
    "seg":   "seg",
}

# =============================================================================
# BraTS 2023 Segmentation Labels
# =============================================================================
LABEL_NAMES = {0: "Background", 1: "NCR", 2: "ED", 3: "ET"}
NUM_CLASSES = 4

REGION_LABELS = {
    "WT": [1, 2, 3],
    "TC": [1, 3],
    "ET": [3],
}

# =============================================================================
# Model & Training Hyperparameters — DGX B200 MIG (23 GB VRAM, 32 GB RAM)
# =============================================================================
TRAIN_RATIO       = 0.80
VAL_RATIO         = 0.10
TEST_RATIO        = 0.10
MAX_PATIENTS      = None

PATCH_SIZE        = (128, 128, 128)
PATCH_OVERLAP     = (32, 32, 32)
NUM_PATCHES_TRAIN = 2

IN_CHANNELS       = 4
OUT_CHANNELS      = 4
INIT_FEATURES     = 32

BATCH_SIZE        = 4
NUM_EPOCHS        = 100
LEARNING_RATE     = 1e-4
WEIGHT_DECAY      = 1e-5
LR_PATIENCE       = 8
LR_FACTOR         = 0.5
EARLY_STOP_PATIENCE = 20
NUM_WORKERS       = 3
PIN_MEMORY        = True
USE_AMP           = True

N_FOLDS           = 3
FOLD              = 0

LOSS_DICE_WEIGHT  = 0.5
LOSS_CE_WEIGHT    = 0.5

RANDOM_SEED       = 42

QUICK_TEST        = False
QUICK_TEST_N      = 10

# =============================================================================
# Device
# =============================================================================
import torch
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# =============================================================================
# Auto-create output directories
# =============================================================================
for _dir in [CHECKPOINT_DIR, RESULTS_DIR, LOGS_DIR]:
    os.makedirs(_dir, exist_ok=True)
