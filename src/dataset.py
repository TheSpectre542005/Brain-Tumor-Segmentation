"""
dataset.py — BraTS 2023 GLI Dataset class
Handles file discovery, loading, patch extraction, and augmentation.
"""

import os
import re
import glob
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from sklearn.model_selection import KFold

import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.config import (
    MODALITIES, PATCH_SIZE, NUM_PATCHES_TRAIN, RANDOM_SEED,
    TRAIN_RATIO, VAL_RATIO, N_FOLDS, MAX_PATIENTS, QUICK_TEST, QUICK_TEST_N,
    BATCH_SIZE, NUM_WORKERS
)


# ---------------------------------------------------------------------------
# Helper: discover all patient folders in BraTS 2023 GLI format
# ---------------------------------------------------------------------------

def get_patient_folders(dataset_path: str) -> List[str]:
    """
    Returns sorted list of patient folder paths.
    Expected pattern: BraTS-GLI-XXXXX-YYY/
    """
    root = Path(dataset_path)
    folders = sorted([
        str(p) for p in root.iterdir()
        if p.is_dir() and re.match(r"BraTS-GLI-\d{5}-\d{3}", p.name)
    ])
    if not folders:
        raise FileNotFoundError(
            f"No BraTS-GLI patient folders found in: {dataset_path}\n"
            f"Expected pattern: BraTS-GLI-XXXXX-YYY/"
        )
    return folders


def get_file_paths(patient_folder: str) -> Dict[str, str]:
    """
    Given a patient folder, returns dict of {modality_key: filepath}.
    E.g. {'flair': '...t2f.nii.gz', 't1': '...t1n.nii.gz', ...}
    """
    folder = Path(patient_folder)
    paths = {}
    for key, suffix in MODALITIES.items():
        matches = list(folder.glob(f"*{suffix}.nii.gz"))
        if not matches:
            raise FileNotFoundError(f"Missing {key} ({suffix}) in {patient_folder}")
        paths[key] = str(matches[0])
    return paths


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def load_volume(filepath: str) -> np.ndarray:
    """Load a .nii.gz file and return as float32 numpy array."""
    img = nib.load(filepath)
    return img.get_fdata(dtype=np.float32)


def normalize_volume(vol: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Z-score normalization within brain mask (non-zero voxels).
    If no mask provided, uses non-zero voxels of the volume itself.
    """
    if mask is None:
        mask = vol > 0
    if mask.sum() == 0:
        return vol
    mean = vol[mask].mean()
    std  = vol[mask].std()
    if std < 1e-8:
        return vol
    return (vol - mean) / (std + 1e-8)


def remap_labels_brats2023(seg: np.ndarray) -> np.ndarray:
    """
    BraTS 2023 labels are already 0,1,2,3 — no remapping needed.
    (BraTS 2021 used 0,1,2,4 — label 4 needed remapping to 3)
    This function is a no-op but kept for clarity / future compatibility.
    """
    return seg.astype(np.int64)


def load_patient(patient_folder: str, has_seg: bool = True
                 ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load all 4 modalities + segmentation for one patient.

    Returns:
        image : shape (4, H, W, D) — stacked modalities, normalized
        seg   : shape (H, W, D)    — label map (None for validation set)
    """
    paths = get_file_paths(patient_folder)

    # Load and normalize each modality
    brain_mask = None
    modality_vols = []
    for key in ["flair", "t1", "t1ce", "t2"]:
        vol = load_volume(paths[key])
        if brain_mask is None:
            brain_mask = vol > 0  # brain mask from FLAIR
        vol = normalize_volume(vol, brain_mask)
        modality_vols.append(vol)

    image = np.stack(modality_vols, axis=0)  # (4, H, W, D)

    seg = None
    if has_seg and "seg" in paths:
        seg = remap_labels_brats2023(load_volume(paths["seg"]))

    return image, seg


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------

def extract_random_patch(image: np.ndarray, seg: np.ndarray,
                          patch_size: Tuple[int, ...] = PATCH_SIZE,
                          foreground_prob: float = 0.67
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a random patch. With probability `foreground_prob`, the patch
    center is sampled from a foreground (non-background) voxel — this
    ensures the model sees tumor in most patches.
    """
    _, H, W, D = image.shape
    ph, pw, pd = patch_size

    # Sample center
    if np.random.random() < foreground_prob:
        fg_coords = np.argwhere(seg > 0)
        if len(fg_coords) > 0:
            center = fg_coords[np.random.randint(len(fg_coords))]
        else:
            center = np.array([H // 2, W // 2, D // 2])
    else:
        center = np.array([
            np.random.randint(ph // 2, H - ph // 2),
            np.random.randint(pw // 2, W - pw // 2),
            np.random.randint(pd // 2, D - pd // 2),
        ])

    # Clip center so patch stays within volume
    ch = int(np.clip(center[0], ph // 2, H - ph // 2))
    cw = int(np.clip(center[1], pw // 2, W - pw // 2))
    cd = int(np.clip(center[2], pd // 2, D - pd // 2))

    sh, sw, sd = ch - ph // 2, cw - pw // 2, cd - pd // 2
    img_patch = image[:, sh:sh+ph, sw:sw+pw, sd:sd+pd]
    seg_patch = seg[sh:sh+ph, sw:sw+pw, sd:sd+pd]

    return img_patch, seg_patch


# ---------------------------------------------------------------------------
# Data augmentation (applied on-the-fly)
# ---------------------------------------------------------------------------

def augment_patch(image: np.ndarray, seg: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple but effective augmentation suite for 3D medical images.
    All ops applied with 50% probability.
    """
    # Random flip along each axis
    for axis in [1, 2, 3]:
        if np.random.random() > 0.5:
            image = np.flip(image, axis=axis).copy()
            seg   = np.flip(seg,   axis=axis - 1).copy()

    # Random 90° rotation in axial plane
    if np.random.random() > 0.5:
        k = np.random.choice([1, 2, 3])
        image = np.rot90(image, k=k, axes=(1, 2)).copy()
        seg   = np.rot90(seg,   k=k, axes=(0, 1)).copy()

    # Gaussian noise
    if np.random.random() > 0.5:
        noise = np.random.normal(0, 0.05, image.shape).astype(np.float32)
        image = (image + noise).astype(np.float32)

    # Random intensity scaling per modality
    if np.random.random() > 0.5:
        scale = np.random.uniform(0.9, 1.1, (image.shape[0], 1, 1, 1)).astype(np.float32)
        image = (image * scale).astype(np.float32)

    return image, seg


# ---------------------------------------------------------------------------
# PyTorch Dataset classes
# ---------------------------------------------------------------------------

class BraTS2023Dataset(Dataset):
    """
    Training/validation dataset.
    Loads volumes lazily (on-demand) to prevent OOM on 32 GB RAM.
    With 1,251 patients each ~100 MB uncompressed, pre-loading would need
    ~125 GB RAM — way beyond the 32 GB allocation.
    """

    def __init__(self, patient_folders: List[str],
                 patch_size: Tuple[int, ...] = PATCH_SIZE,
                 num_patches: int = NUM_PATCHES_TRAIN,
                 augment: bool = True):
        self.patient_folders = patient_folders
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.augment = augment
        print(f"Dataset initialised: {len(patient_folders)} patients "
              f"({'train+aug' if augment else 'val'}) — lazy loading enabled")

    def __len__(self):
        return len(self.patient_folders) * self.num_patches

    def __getitem__(self, idx):
        patient_idx = idx // self.num_patches
        # Load patient volume on demand (not pre-loaded)
        image, seg = load_patient(self.patient_folders[patient_idx], has_seg=True)
        img_patch, seg_patch = extract_random_patch(image, seg, self.patch_size)

        if self.augment:
            img_patch, seg_patch = augment_patch(img_patch, seg_patch)

        return (
            torch.from_numpy(img_patch.astype(np.float32)),
            torch.from_numpy(seg_patch.astype(np.int64))
        )


class BraTS2023InferenceDataset(Dataset):
    """
    Inference dataset — loads full volumes (no patching, no augmentation).
    Used for evaluation / visualisation.
    """

    def __init__(self, patient_folders: List[str], has_seg: bool = True):
        self.patient_folders = patient_folders
        self.has_seg = has_seg

    def __len__(self):
        return len(self.patient_folders)

    def __getitem__(self, idx):
        folder = self.patient_folders[idx]
        image, seg = load_patient(folder, has_seg=self.has_seg)

        data = {"image": torch.from_numpy(image.astype(np.float32)),
                "folder": folder}
        if seg is not None:
            data["seg"] = torch.from_numpy(seg.astype(np.int64))
        return data


# ---------------------------------------------------------------------------
# Dataloader factory
# ---------------------------------------------------------------------------

def get_dataloaders(dataset_path: str, batch_size: int = 2,
                    num_workers: int = 4, fold: int = 0, n_folds: int = 3,
                    max_patients: Optional[int] = None,
                    quick_test: bool = False, quick_test_n: int = 10
                    ) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Discover patients, split into train/val/test using SUBJECT-LEVEL grouping
    (preventing multi-scan patient leakage), return DataLoaders.

    Returns: train_loader, val_loader, test_loader, test_folders
    """
    np.random.seed(RANDOM_SEED)

    folders = get_patient_folders(dataset_path)

    if quick_test:
        folders = folders[:quick_test_n]
        print(f"[QUICK TEST MODE] Using {len(folders)} scans.")
    elif max_patients and max_patients < len(folders):
        folders = folders[:max_patients]
        print(f"[INFO] Using {len(folders)} / {len(folders)} scans.")

    # ── Group by unique Subject ID (BraTS-GLI-XXXXX) to PREVENT DATA LEAKAGE ──
    # Some subjects have multiple scans (e.g. baseline -000 and follow-up -001).
    # All scans from the same subject MUST stay in the same split.
    from collections import defaultdict
    subject_to_folders = defaultdict(list)
    for f in folders:
        match = re.search(r"(BraTS-GLI-\d{5})", Path(f).name)
        subj_id = match.group(1) if match else Path(f).name
        subject_to_folders[subj_id].append(f)

    unique_subjects = sorted(list(subject_to_folders.keys()))
    np.random.shuffle(unique_subjects)

    # Hold out 10% of unique subjects for final test set
    n_test_subj = max(1, int(len(unique_subjects) * 0.10))
    test_subj = unique_subjects[-n_test_subj:]
    trainval_subj = unique_subjects[:-n_test_subj]

    test_folders = [f for s in test_subj for f in subject_to_folders[s]]

    # K-Fold on unique training/validation subjects
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    splits = list(kf.split(trainval_subj))
    train_subj_idx, val_subj_idx = splits[fold]

    train_folders = [f for i in train_subj_idx for f in subject_to_folders[trainval_subj[i]]]
    val_folders   = [f for i in val_subj_idx   for f in subject_to_folders[trainval_subj[i]]]

    # Verify zero leakage across splits
    train_s = set(re.search(r"(BraTS-GLI-\d{5})", Path(f).name).group(1) for f in train_folders)
    val_s   = set(re.search(r"(BraTS-GLI-\d{5})", Path(f).name).group(1) for f in val_folders)
    test_s  = set(re.search(r"(BraTS-GLI-\d{5})", Path(f).name).group(1) for f in test_folders)
    assert len(train_s.intersection(val_s)) == 0, "Data Leakage detected between Train and Val!"
    assert len(train_s.intersection(test_s)) == 0, "Data Leakage detected between Train and Test!"
    assert len(val_s.intersection(test_s)) == 0, "Data Leakage detected between Val and Test!"

    print(f"Fold {fold}/{n_folds} (Subject-Grouped, 0 Leakage): "
          f"Train={len(train_folders)} scans ({len(train_s)} subjects), "
          f"Val={len(val_folders)} scans ({len(val_s)} subjects), "
          f"Test={len(test_folders)} scans ({len(test_s)} subjects)")

    train_ds = BraTS2023Dataset(train_folders, augment=True)
    val_ds   = BraTS2023Dataset(val_folders,   augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    # Test loader uses InferenceDataset (full volumes)
    test_ds     = BraTS2023InferenceDataset(test_folders, has_seg=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, test_folders
