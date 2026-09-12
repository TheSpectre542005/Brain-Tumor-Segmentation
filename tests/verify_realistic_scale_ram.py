import os
import sys
import time
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
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.dataset import BraTS2023Dataset

def get_tree_ram_mb():
    """Calculates total RSS memory of current process plus all spawned worker children."""
    parent = psutil.Process(os.getpid())
    total_rss = parent.memory_info().rss
    try:
        children = parent.children(recursive=True)
        for child in children:
            if child.is_running():
                try:
                    total_rss += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
    except Exception:
        pass
    return total_rss / (1024 * 1024)

def main():
    print("=" * 75)
    print("  Real-Scale BraTS (240x240x155) RAM & Lazy-Loading Benchmark")
    print("=" * 75)

    H, W, D = 240, 240, 155
    patch_size = (96, 96, 96)
    batch_size = 2
    num_workers = 2 if sys.platform == "win32" else 4
    num_patches_train = 4
    num_patients = 12

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        print(f"Creating {num_patients} full-scale BraTS patient scans ({H}x{W}x{D}, 4 modalities)...", flush=True)
        
        # Create realistic synthetic brain volume
        affine = np.eye(4)
        vol_flair = np.zeros((H, W, D), dtype=np.float32)
        grid_y, grid_x, grid_z = np.ogrid[:H, :W, :D]
        brain_mask = (((grid_y - 120)/80)**2 + ((grid_x - 120)/80)**2 + ((grid_z - 77)/55)**2) <= 1.0
        vol_flair[brain_mask] = np.random.uniform(0.2, 1.0, size=brain_mask.sum()).astype(np.float32)
        
        seg_vol = np.zeros((H, W, D), dtype=np.int16)
        tumor_mask = (((grid_y - 130)/25)**2 + ((grid_x - 120)/25)**2 + ((grid_z - 80)/20)**2) <= 1.0
        seg_vol[tumor_mask] = 3  # ET
        
        nii_flair = nib.Nifti1Image(vol_flair, affine)
        nii_seg   = nib.Nifti1Image(seg_vol, affine)
        
        folders = []
        for i in range(num_patients):
            sdir = tmp_path / f"BraTS-GLI-{i:05d}-000"
            sdir.mkdir(parents=True, exist_ok=True)
            folders.append(str(sdir))
            
            nib.save(nii_flair, str(sdir / f"{sdir.name}-t2f.nii.gz"))
            nib.save(nii_flair, str(sdir / f"{sdir.name}-t1n.nii.gz"))
            nib.save(nii_flair, str(sdir / f"{sdir.name}-t1c.nii.gz"))
            nib.save(nii_flair, str(sdir / f"{sdir.name}-t2w.nii.gz"))
            nib.save(nii_seg,   str(sdir / f"{sdir.name}-seg.nii.gz"))

        print(f"Generated {num_patients} full-scale NIfTI directories ({num_patients * 5} NIfTI files total).", flush=True)
        
        # 1. Baseline Memory
        ram_baseline = get_tree_ram_mb()
        print(f"\n[1] Baseline Process RAM         : {ram_baseline:8.2f} MB", flush=True)

        # 2. Dataset Instantiation
        dataset = BraTS2023Dataset(
            patient_folders=folders,
            patch_size=patch_size,
            num_patches=num_patches_train,
            foreground_prob=0.67,
            augment=True
        )
        ram_after_ds = get_tree_ram_mb()
        print(f"[2] After Dataset Init (N={len(dataset)} samples): {ram_after_ds:8.2f} MB  (Δ = +{ram_after_ds - ram_baseline:.2f} MB)", flush=True)

        # 3. DataLoader Instantiation
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False
        )
        ram_after_loader = get_tree_ram_mb()
        print(f"[3] After DataLoader Init        : {ram_after_loader:8.2f} MB  (Workers: {num_workers})", flush=True)

        # 4. Iterating through 20 full batches
        print(f"\n[4] Streaming 20 Full-Scale Batches (Batch size: {batch_size}, Patch: {patch_size})...", flush=True)
        batch_ram_readings = []
        t0 = time.time()
        
        batch_count = 0
        for epoch in range(3):
            for img_batch, seg_batch in loader:
                batch_count += 1
                cur_ram = get_tree_ram_mb()
                batch_ram_readings.append(cur_ram)
                if batch_count % 5 == 0 or batch_count == 1:
                    print(f"    Batch {batch_count:02d}/20 -> Total Tree RAM: {cur_ram:8.2f} MB ({cur_ram/1024:.2f} GB) | Batch Tensor: {tuple(img_batch.shape)}", flush=True)
                if batch_count >= 20:
                    break
            if batch_count >= 20:
                break

        elapsed = time.time() - t0
        peak_ram = max(batch_ram_readings)
        min_ram  = min(batch_ram_readings)
        mean_ram = float(np.mean(batch_ram_readings))

        print("\n" + "=" * 75, flush=True)
        print("  📊 Realistic-Scale Memory Profile & Server Safety Analysis", flush=True)
        print("=" * 75, flush=True)
        print(f"  • Baseline Memory            : {ram_baseline:8.2f} MB ({ram_baseline / 1024:.2f} GB)", flush=True)
        print(f"  • Peak Total Process Tree RAM: {peak_ram:8.2f} MB ({peak_ram / 1024:.2f} GB)", flush=True)
        print(f"  • Average Batch RAM          : {mean_ram:8.2f} MB ({mean_ram / 1024:.2f} GB)", flush=True)
        print(f"  • Target Server RAM Limit    : 32768.00 MB (32.00 GB)", flush=True)
        print(f"  • Server RAM Utilization     : {(peak_ram / 32768.0) * 100:.2f}% of 32GB budget", flush=True)
        print(f"  • Remaining Free RAM Buffer  : {(32768.0 - peak_ram) / 1024:.2f} GB (~{100 - (peak_ram / 32768.0)*100:.1f}% safety margin)", flush=True)
        print(f"  • Throughput                 : {batch_count / elapsed:.2f} batches/sec ({batch_count * batch_size / elapsed:.2f} patches/sec)", flush=True)
        print("=" * 75, flush=True)

        assert peak_ram < 16384, f"CRITICAL: Peak RAM ({peak_ram:.2f} MB) exceeded 50% of total 32GB budget!"
        print("STATUS: REALISTIC-SCALE BRAIN TUMOR PIPELINE RAM SAFETY VERIFIED (100% PASS).", flush=True)

if __name__ == "__main__":
    main()
