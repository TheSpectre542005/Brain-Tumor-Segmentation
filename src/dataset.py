"""
dataset.py — BraTS 2023 GLI Dataset and Dataloaders
Handles patient discovery, subject-level zero-leakage splits, lazy loading,
on-the-fly foreground-biased patch extraction, and data augmentation.
"""

import os
import re
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union
from collections import defaultdict

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold

from src.config import (
    MODALITIES, PATCH_SIZE, NUM_PATCHES_TRAIN, FOREGROUND_PROB,
    RANDOM_SEED, NUM_WORKERS, PIN_MEMORY, BATCH_SIZE
)


# =============================================================================
# 1. File Discovery & Path Resolution
# =============================================================================

def get_patient_folders(dataset_path: Union[str, Path]) -> List[str]:
    """
    Returns sorted list of patient folder paths matching BraTS format.
    Accepts paths with pattern BraTS-GLI-XXXXX-YYY or BraTS2023_XXXXX.
    """
    root = Path(dataset_path)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root path does not exist: {dataset_path}")

    folders = sorted([
        str(p) for p in root.iterdir()
        if p.is_dir() and (re.search(r"BraTS", p.name, re.IGNORECASE) or (p / f"{p.name}-t2f.nii.gz").exists())
    ])

    if not folders:
        # Check if root itself is a flat directory with .nii.gz or subfolders
        sub_dirs = sorted([str(p) for p in root.glob("*") if p.is_dir()])
        if sub_dirs:
            folders = sub_dirs
        else:
            raise FileNotFoundError(
                f"No patient folders found in: {dataset_path}\n"
                f"Expected directory structure containing patient folders (e.g. BraTS-GLI-XXXXX-YYY/)"
            )
    return folders


def get_file_paths(patient_folder: Union[str, Path]) -> Dict[str, str]:
    """
    Given a patient folder, returns dictionary mapping modality key to absolute filepath.
    E.g. {'flair': '...-t2f.nii.gz', 't1': '...-t1n.nii.gz', 't1ce': '...-t1c.nii.gz',
          't2': '...-t2w.nii.gz', 'seg': '...-seg.nii.gz'}
    """
    folder = Path(patient_folder)
    paths: Dict[str, str] = {}

    for key, suffix in MODALITIES.items():
        # Match standard BraTS 2023 suffix e.g. *-t2f.nii.gz or *-seg.nii.gz
        matches = list(folder.glob(f"*{suffix}.nii*"))
        if not matches and key == "flair":
            matches = list(folder.glob("*flair.nii*")) or list(folder.glob("*FLAIR.nii*"))
        elif not matches and key == "t1":
            matches = list(folder.glob("*t1.nii*")) or list(folder.glob("*T1.nii*"))
        elif not matches and key == "t1ce":
            matches = list(folder.glob("*t1ce.nii*")) or list(folder.glob("*T1CE.nii*")) or list(folder.glob("*t1c.nii*"))
        elif not matches and key == "t2":
            matches = list(folder.glob("*t2.nii*")) or list(folder.glob("*T2.nii*"))
        elif not matches and key == "seg":
            matches = list(folder.glob("*seg.nii*")) or list(folder.glob("*SEG.nii*"))

        if matches:
            paths[key] = str(matches[0])
        elif key != "seg":  # seg may be absent in unannotated test sets
            raise FileNotFoundError(f"Missing mandatory modality '{key}' (expected suffix '{suffix}') in {patient_folder}")

    return paths


# =============================================================================
# 2. Loading, Normalization & Label Remapping
# =============================================================================

def load_volume(filepath: Union[str, Path]) -> np.ndarray:
    """Load a .nii / .nii.gz file and return as float32 numpy array."""
    img = nib.load(str(filepath))
    return np.asanyarray(img.dataobj, dtype=np.float32)


def normalize_volume(vol: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Z-score normalization strictly within non-zero brain mask voxels.
    Zero-background voxels remain 0.

    Ensures zero dataset-level normalization leakage: computed independently per scan.
    """
    if mask is None:
        mask = vol > 0
    if mask.sum() == 0:
        return np.zeros_like(vol, dtype=np.float32)

    mean = vol[mask].mean()
    std  = vol[mask].std()
    if std < 1e-8:
        return np.zeros_like(vol, dtype=np.float32)

    normalized = np.zeros_like(vol, dtype=np.float32)
    normalized[mask] = (vol[mask] - mean) / (std + 1e-8)
    return normalized


def remap_labels(seg: np.ndarray) -> np.ndarray:
    """
    Ensures BraTS segmentation labels are strictly contiguous integers in {0, 1, 2, 3}.
    Remaps legacy BraTS 2021 label 4 (Enhancing Tumor) to 3 if present.
    """
    seg = seg.astype(np.int64)
    if (seg == 4).any():
        seg = seg.copy()
        seg[seg == 4] = 3
    return seg

# Backward-compatibility alias for notebooks
remap_labels_brats2023 = remap_labels


def load_patient(patient_folder: Union[str, Path],
                 has_seg: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load all 4 modalities + segmentation for a patient case.

    Returns:
        image : (4, H, W, D) float32 array — [FLAIR, T1, T1ce, T2] normalized
        seg   : (H, W, D) int64 label map in {0, 1, 2, 3}, or None if unavailable
    """
    paths = get_file_paths(patient_folder)

    # 1. Load FLAIR first to establish brain mask
    flair_vol = load_volume(paths["flair"])
    brain_mask = flair_vol > 0

    # 2. Normalize each modality independently within the brain mask
    modality_vols = []
    for key in ["flair", "t1", "t1ce", "t2"]:
        vol = flair_vol if key == "flair" else load_volume(paths[key])
        vol_norm = normalize_volume(vol, brain_mask)
        modality_vols.append(vol_norm)

    image = np.stack(modality_vols, axis=0).astype(np.float32)  # (4, H, W, D)

    seg = None
    if has_seg and "seg" in paths:
        raw_seg = load_volume(paths["seg"])
        seg = remap_labels(raw_seg)

    return image, seg


# =============================================================================
# 3. Patient-Level Splitting (Zero Leakage Guarantee)
# =============================================================================

def extract_subject_id(folder_path: str) -> str:
    """
    Extracts the unique Subject ID (e.g. BraTS-GLI-00001 from BraTS-GLI-00001-000).
    Ensures all temporal/longitudinal scans of the same patient are grouped together.
    """
    name = Path(folder_path).name
    match = re.search(r"(BraTS[^\-_]*-[A-Z]+-\d{5}|BraTS\d{4}_\d{5}|BraTS-GLI-\d{5})", name, re.IGNORECASE)
    if match:
        return match.group(1)
    # Fallback to entire folder name if no multi-scan pattern matched
    return name


def get_subject_splits(folders: List[str],
                       n_folds: int = 3,
                       fold: int = 0,
                       test_ratio: float = 0.10,
                       seed: int = RANDOM_SEED,
                       split_file: Optional[Union[str, Path]] = None
                       ) -> Tuple[List[str], List[str], List[str]]:
    """
    Splits patient scans into train, val, and test splits grouped strictly by Subject ID.

    Guarantees:
      1. All scans belonging to one subject ID stay in exactly one split.
      2. Test set (10% of unique subjects) is held out completely.
      3. K-Fold cross-validation splits remaining subjects without overlap.
      4. Zero subject intersection across (Train, Val, Test).

    Returns:
        train_folders, val_folders, test_folders
    """
    # Group scan paths by unique subject ID
    subject_to_folders: Dict[str, List[str]] = defaultdict(list)
    for f in folders:
        sid = extract_subject_id(f)
        subject_to_folders[sid].append(f)

    unique_subjects = sorted(list(subject_to_folders.keys()))
    rng = np.random.RandomState(seed)
    shuffled_subjects = unique_subjects.copy()
    rng.shuffle(shuffled_subjects)

    # 1. Hold out test subjects
    n_test_subj = max(1, int(len(shuffled_subjects) * test_ratio))
    test_subjects = set(shuffled_subjects[-n_test_subj:])
    trainval_subjects = shuffled_subjects[:-n_test_subj]

    # 2. K-Fold split on train/val subjects
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = list(kf.split(trainval_subjects))
    train_idx, val_idx = splits[fold]

    train_subjects = set(trainval_subjects[i] for i in train_idx)
    val_subjects   = set(trainval_subjects[i] for i in val_idx)

    # Assert ZERO leakage between subject groups
    assert len(train_subjects.intersection(val_subjects)) == 0, "Patient-level leakage between Train and Val!"
    assert len(train_subjects.intersection(test_subjects)) == 0, "Patient-level leakage between Train and Test!"
    assert len(val_subjects.intersection(test_subjects)) == 0, "Patient-level leakage between Val and Test!"

    train_folders = [f for s in train_subjects for f in subject_to_folders[s]]
    val_folders   = [f for s in val_subjects   for f in subject_to_folders[s]]
    test_folders  = [f for s in test_subjects  for f in subject_to_folders[s]]

    # Optional: cache split manifest for exact multi-process reproducibility
    if split_file:
        manifest = {
            "fold": fold,
            "n_folds": n_folds,
            "seed": seed,
            "train_scans": [Path(f).name for f in train_folders],
            "val_scans":   [Path(f).name for f in val_folders],
            "test_scans":  [Path(f).name for f in test_folders],
            "train_subjects": sorted(list(train_subjects)),
            "val_subjects":   sorted(list(val_subjects)),
            "test_subjects":  sorted(list(test_subjects)),
        }
        os.makedirs(Path(split_file).parent, exist_ok=True)
        with open(split_file, "w") as fp:
            json.dump(manifest, fp, indent=2)

    return train_folders, val_folders, test_folders


# =============================================================================
# 4. Patch Extraction & Data Augmentation
# =============================================================================

def extract_random_patch(image: np.ndarray,
                         seg: np.ndarray,
                         patch_size: Tuple[int, int, int] = PATCH_SIZE,
                         foreground_prob: float = FOREGROUND_PROB
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts a 3D volumetric patch from full 4D image (4, H, W, D) and 3D seg (H, W, D).
    Uses foreground-biased sampling: with probability `foreground_prob`, centers patch
    on a tumor voxel (seg > 0). Used exclusively for training.
    """
    _, H, W, D = image.shape
    ph, pw, pd = patch_size

    # Ensure volume is large enough for patch size
    if H < ph or W < pw or D < pd:
        pad_h = max(0, ph - H)
        pad_w = max(0, pw - W)
        pad_d = max(0, pd - D)
        image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w), (0, pad_d)), mode="constant")
        seg   = np.pad(seg,   ((0, pad_h), (0, pad_w), (0, pad_d)), mode="constant")
        _, H, W, D = image.shape

    sample_foreground = (np.random.random() < foreground_prob) and (seg > 0).any()
    if sample_foreground:
        fg_coords = np.argwhere(seg > 0)
        center = fg_coords[np.random.randint(len(fg_coords))]
    else:
        center = np.array([
            np.random.randint(ph // 2, max(ph // 2 + 1, H - ph // 2)),
            np.random.randint(pw // 2, max(pw // 2 + 1, W - pw // 2)),
            np.random.randint(pd // 2, max(pd // 2 + 1, D - pd // 2)),
        ])

    ch = int(np.clip(center[0], ph // 2, H - (ph - ph // 2)))
    cw = int(np.clip(center[1], pw // 2, W - (pw - pw // 2)))
    cd = int(np.clip(center[2], pd // 2, D - (pd - pd // 2)))

    sh, sw, sd = ch - ph // 2, cw - pw // 2, cd - pd // 2
    img_patch = image[:, sh:sh + ph, sw:sw + pw, sd:sd + pd]
    seg_patch = seg[sh:sh + ph, sw:sw + pw, sd:sd + pd]

    return img_patch, seg_patch


def augment_patch(image: np.ndarray, seg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    On-the-fly 3D data augmentation suite (applied strictly to training patches):
      1. Random 3D spatial flips along axial, coronal, sagittal axes (50% prob each)
      2. Random 90° rotation in axial plane (50% prob)
      3. Additive Gaussian noise (50% prob)
      4. Per-channel random intensity scaling (50% prob)
    """
    image = image.copy()
    seg   = seg.copy()

    # 1. Random axis flips
    for axis in [1, 2, 3]:
        if np.random.random() > 0.5:
            image = np.flip(image, axis=axis).copy()
            seg   = np.flip(seg,   axis=axis - 1).copy()

    # 2. Random 90° rotation in axial plane
    if np.random.random() > 0.5:
        k = int(np.random.choice([1, 2, 3]))
        image = np.rot90(image, k=k, axes=(1, 2)).copy()
        seg   = np.rot90(seg,   k=k, axes=(0, 1)).copy()

    # 3. Subtle Gaussian noise on image channels
    if np.random.random() > 0.5:
        noise = np.random.normal(0.0, 0.05, size=image.shape).astype(np.float32)
        image = image + noise

    # 4. Multiplicative intensity scaling per modality
    if np.random.random() > 0.5:
        scales = np.random.uniform(0.9, 1.1, size=(image.shape[0], 1, 1, 1)).astype(np.float32)
        image = image * scales

    return image, seg


# =============================================================================
# 5. PyTorch Dataset Classes
# =============================================================================

class BraTS2023Dataset(Dataset):
    """
    Training Dataset class for BraTS 2023 GLI.
    Loads patient scans lazily on-demand from disk (preventing RAM exhaustion).
    Extracts random patches on-the-fly with augmentation.
    """

    def __init__(self,
                 patient_folders: List[str],
                 patch_size: Tuple[int, int, int] = PATCH_SIZE,
                 num_patches: int = NUM_PATCHES_TRAIN,
                 foreground_prob: float = FOREGROUND_PROB,
                 augment: bool = True):
        self.patient_folders = patient_folders
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.foreground_prob = foreground_prob
        self.augment = augment

    def __len__(self) -> int:
        return len(self.patient_folders) * self.num_patches

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        patient_idx = idx // self.num_patches
        folder = self.patient_folders[patient_idx]

        # Lazy loading on-demand per sample
        image, seg = load_patient(folder, has_seg=True)
        if seg is None:
            raise ValueError(f"Ground-truth segmentation missing for training sample: {folder}")

        img_patch, seg_patch = extract_random_patch(
            image, seg, self.patch_size, self.foreground_prob
        )

        if self.augment:
            img_patch, seg_patch = augment_patch(img_patch, seg_patch)

        return (
            torch.from_numpy(img_patch.astype(np.float32)),
            torch.from_numpy(seg_patch.astype(np.int64))
        )


class BraTS2023InferenceDataset(Dataset):
    """
    Full-volume inference dataset for validation / testing.
    Loads unaugmented 3D volumes with full spatial geometry for sliding-window inference.
    """

    def __init__(self, patient_folders: List[str], has_seg: bool = True):
        self.patient_folders = patient_folders
        self.has_seg = has_seg

    def __len__(self) -> int:
        return len(self.patient_folders)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        folder = self.patient_folders[idx]
        image, seg = load_patient(folder, has_seg=self.has_seg)

        data = {
            "image": torch.from_numpy(image.astype(np.float32)),
            "folder": str(folder),
            "patient_id": Path(folder).name,
        }
        if seg is not None:
            data["seg"] = torch.from_numpy(seg.astype(np.int64))
        return data


# =============================================================================
# 6. DataLoader Factory
# =============================================================================

def seed_worker(worker_id: int):
    """Ensure worker processes inherit independent deterministic random seeds."""
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)


def get_dataloaders(dataset_path: Union[str, Path],
                    batch_size: int = BATCH_SIZE,
                    num_workers: int = NUM_WORKERS,
                    pin_memory: bool = PIN_MEMORY,
                    fold: int = 0,
                    n_folds: int = 3,
                    max_patients: Optional[int] = None,
                    quick_test: bool = False,
                    quick_test_n: int = 10,
                    seed: int = RANDOM_SEED
                    ) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Full pipeline dataloader factory:
      1. Discovers patients and applies max_patients/quick_test subsets.
      2. Generates zero-leakage Subject-level train/val/test splits.
      3. Returns train_loader (patch-based), val_loader (full-volume),
         test_loader (full-volume), and test_folders.
    """
    folders = get_patient_folders(dataset_path)

    if quick_test:
        folders = folders[:quick_test_n]
        print(f"[QUICK TEST MODE] Limited to {len(folders)} scans.")
    elif max_patients and max_patients < len(folders):
        folders = folders[:max_patients]
        print(f"[INFO] Limited to {len(folders)} scans (MAX_PATIENTS).")

    train_folders, val_folders, test_folders = get_subject_splits(
        folders, n_folds=n_folds, fold=fold, seed=seed
    )

    print(f"\n📂 Dataset Splits (Fold {fold}/{n_folds} — Subject-Grouped, 0 Leakage):")
    print(f"   Train : {len(train_folders):4d} scans ({len(set(extract_subject_id(f) for f in train_folders))} subjects)")
    print(f"   Val   : {len(val_folders):4d} scans ({len(set(extract_subject_id(f) for f in val_folders))} subjects)")
    print(f"   Test  : {len(test_folders):4d} scans ({len(set(extract_subject_id(f) for f in test_folders))} subjects)\n")

    train_ds = BraTS2023Dataset(train_folders, augment=True)
    val_ds   = BraTS2023InferenceDataset(val_folders, has_seg=True)
    test_ds  = BraTS2023InferenceDataset(test_folders, has_seg=True)

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        worker_init_fn=seed_worker, generator=g
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=min(2, num_workers), pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=min(2, num_workers), pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader, test_folders
