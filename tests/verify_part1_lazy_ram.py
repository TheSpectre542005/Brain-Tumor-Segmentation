import os
import sys
import tempfile
import psutil
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import DataLoader

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.dataset import BraTS2023Dataset

def get_ram_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

print("=" * 65)
print("  Part 1 - Check 3: Lazy Loading RAM Verification (50 Subjects)")
print("=" * 65)

with tempfile.TemporaryDirectory() as tmpdir:
    tmp_path = Path(tmpdir)
    print(f"Creating 50 synthetic subject directories with NIfTI files in {tmp_path.name}...")
    
    # Create 50 subject folders with lightweight NIfTI files
    folders = []
    affine = np.eye(4)
    dummy_vol = np.zeros((64, 64, 64), dtype=np.float32)
    dummy_vol[16:48, 16:48, 16:48] = 1.0  # brain mask
    dummy_seg = np.zeros((64, 64, 64), dtype=np.int16)
    dummy_seg[24:40, 24:40, 24:40] = 3    # tumor
    
    img_nii = nib.Nifti1Image(dummy_vol, affine)
    seg_nii = nib.Nifti1Image(dummy_seg, affine)

    for i in range(50):
        sdir = tmp_path / f"BraTS-GLI-{i:05d}-000"
        sdir.mkdir(parents=True, exist_ok=True)
        folders.append(str(sdir))
        
        # Save modalities
        nib.save(img_nii, str(sdir / f"{sdir.name}-t2f.nii.gz"))
        nib.save(img_nii, str(sdir / f"{sdir.name}-t1n.nii.gz"))
        nib.save(img_nii, str(sdir / f"{sdir.name}-t1c.nii.gz"))
        nib.save(img_nii, str(sdir / f"{sdir.name}-t2w.nii.gz"))
        nib.save(seg_nii, str(sdir / f"{sdir.name}-seg.nii.gz"))

    ram_initial = get_ram_mb()
    print(f"\n[1] Baseline Process RAM : {ram_initial:8.2f} MB")

    # Instantiate BraTS2023Dataset with 50 subjects x 4 patches = 200 samples
    dataset = BraTS2023Dataset(
        patient_folders=folders,
        patch_size=(32, 32, 32),
        num_patches=4,
        foreground_prob=0.8,
        augment=True
    )
    ram_after_init = get_ram_mb()
    delta_init = ram_after_init - ram_initial
    print(f"[2] After Dataset Init   : {ram_after_init:8.2f} MB  (Δ = +{delta_init:.2f} MB for 50 subjects / 200 samples)")

    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    ram_after_loader = get_ram_mb()
    print(f"[3] After DataLoader Init: {ram_after_loader:8.2f} MB  (Δ = +{ram_after_loader - ram_after_init:.2f} MB)")

    # Fetch 1 batch
    batch_img, batch_seg = next(iter(loader))
    ram_after_batch1 = get_ram_mb()
    print(f"[4] After Batch 1 Fetch  : {ram_after_batch1:8.2f} MB  (Tensor: {tuple(batch_img.shape)}, Seg: {tuple(batch_seg.shape)})")

    # Fetch 10 batches
    for i, (b_img, b_seg) in enumerate(loader):
        if i >= 10:
            break
    ram_after_10_batches = get_ram_mb()
    print(f"[5] After 10 Batches Fetch: {ram_after_10_batches:8.2f} MB  (Δ = +{ram_after_10_batches - ram_after_batch1:.2f} MB)")

print("\n--- Memory Overhead Summary ---")
print(f"Dataset Init Memory Overhead: {delta_init:.2f} MB (Only file path strings stored in RAM, volumes loaded lazily on-demand)")
assert delta_init < 50.0, f"RAM overhead ({delta_init:.2f} MB) too high for lazy dataset!"
print("STATUS: ZERO-PRELOAD LAZY LOADING RAM EFFICIENCY CONFIRMED.")
