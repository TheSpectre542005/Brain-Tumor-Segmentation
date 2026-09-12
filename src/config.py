"""
config.py — Centralized Configuration for BraTS 2023 GLI Brain Tumor Segmentation
Handles paths, labels, sub-region definitions, network architecture, and training hyperparameters.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple
import torch

# =============================================================================
# 📁 Paths (Configurable via Environment Variables or direct assignment)
# =============================================================================
DATASET_PATH = os.environ.get(
    "BRATS_DATASET_PATH",
    "<UPDATE THIS TO YOUR ACTUAL DATASET LOCATION ON THE SERVER — e.g. /GPFS_CLUSTER/students/student02/Dataset_final>"
)
VALIDATION_PATH = os.environ.get(
    "BRATS_VALIDATION_PATH",
    "<UPDATE THIS TO YOUR ACTUAL VALIDATION DATASET LOCATION ON THE SERVER>"
)
PREPROCESSED_PATH = os.environ.get(
    "BRATS_PREPROCESSED_PATH",
    "./data/preprocessed"
)
CHECKPOINT_DIR = os.environ.get("BRATS_CHECKPOINT_DIR", "./checkpoints")
RESULTS_DIR    = os.environ.get("BRATS_RESULTS_DIR", "./results")
LOGS_DIR       = os.environ.get("BRATS_LOGS_DIR", "./logs")


def ensure_directories(dirs: List[str] = None) -> None:
    """Safely create required output directories only when invoked (no import side-effects)."""
    if dirs is None:
        dirs = [CHECKPOINT_DIR, RESULTS_DIR, LOGS_DIR]
    for d in dirs:
        if d and not str(d).startswith("<UPDATE") and not str(d).startswith("/home/yourname"):
            os.makedirs(d, exist_ok=True)


# =============================================================================
# 📂 BraTS 2023 GLI Modalities & File Conventions
# =============================================================================
# Pattern: BraTS-GLI-{XXXXX}-{YYY}-{modality}.nii.gz
MODALITIES = {
    "flair": "t2f",   # T2-FLAIR
    "t1":    "t1n",   # T1 native
    "t1ce":  "t1c",   # T1 contrast-enhanced
    "t2":    "t2w",   # T2-weighted
    "seg":   "seg",   # Ground truth segmentation mask
}

# =============================================================================
# 🏷️ BraTS Segmentation Labels & Clinical Sub-Regions
# =============================================================================
# Label 0 → Background (healthy tissue / air)
# Label 1 → Necrotic Tumor Core (NCR)
# Label 2 → Peritumoral Edema (ED)
# Label 3 → Enhancing Tumor (ET)  [Note: BraTS 2021 used 4, remapped to 3]
LABEL_NAMES: Dict[int, str] = {
    0: "Background",
    1: "Necrotic Core (NCR)",
    2: "Peritumoral Edema (ED)",
    3: "Enhancing Tumor (ET)",
}
NUM_CLASSES: int = 4

# BraTS evaluation sub-regions:
#  WT (Whole Tumor)  = labels 1 + 2 + 3 (all tumor regions)
#  TC (Tumor Core)   = labels 1 + 3 (resectable core: NCR + ET)
#  ET (Enhancing)    = label 3 (active enhancing rim)
REGION_LABELS: Dict[str, List[int]] = {
    "WT": [1, 2, 3],
    "TC": [1, 3],
    "ET": [3],
}

# =============================================================================
# 🧠 Model & Hyperparameter Settings (Tuned for 23GB MIG Slice / B200 / 8-core CPU)
# =============================================================================
# -- Network dimensions --
IN_CHANNELS: int   = 4               # 4 MRI sequences (FLAIR, T1, T1ce, T2)
OUT_CHANNELS: int  = 4               # 4 output classes (BG, NCR, ED, ET)
INIT_FEATURES: int = 32              # Base feature channels in encoder

# -- Patch extraction & Inference --
PATCH_SIZE: Tuple[int, int, int]    = (96, 96, 96)   # 96³ 3D volumetric patch
PATCH_OVERLAP: Tuple[int, int, int] = (48, 48, 48)   # 50% overlap for sliding window
NUM_PATCHES_TRAIN: int              = 2              # Random patches extracted per volume per epoch
FOREGROUND_PROB: float              = 0.67           # Probability of tumor-centered patch sampling in training

# -- Dataset Splitting --
TRAIN_RATIO: float  = 0.80
VAL_RATIO: float    = 0.10
TEST_RATIO: float   = 0.10
N_FOLDS: int        = 3
FOLD: int           = 0
MAX_PATIENTS: int   = 600

# -- Optimization & Training --
BATCH_SIZE: int          = 2
NUM_EPOCHS: int          = 150
LEARNING_RATE: float     = 1e-4
WEIGHT_DECAY: float      = 1e-5
LR_PATIENCE: int         = 10        # ReduceLROnPlateau patience
LR_FACTOR: float         = 0.5
EARLY_STOP_PATIENCE: int = 25

# -- Hardware & Dataloader (Optimized for 8-core CPU, 32GB RAM, 23GB VRAM) --
NUM_WORKERS: int         = 4         # Safe for 8-core CPU allocation
PIN_MEMORY: bool         = True
USE_AMP: bool            = True      # Mixed precision (FP16/BF16) with GradScaler

# -- Loss Function Weights --
LOSS_DICE_WEIGHT: float  = 0.5
LOSS_CE_WEIGHT: float    = 0.5
FOCAL_GAMMA: float       = 2.0

# -- Reproducibility --
RANDOM_SEED: int         = 42

# -- Quick Test Mode --
QUICK_TEST: bool         = False
QUICK_TEST_N: int        = 10


def get_device() -> torch.device:
    """Return active compute device without side effects."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


DEVICE = get_device()
