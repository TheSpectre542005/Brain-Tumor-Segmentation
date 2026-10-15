#!/usr/bin/env python3
"""
dgx_healthcheck.py — Run this FIRST on the DGX before launching any training.
Takes ~5 minutes. Catches every common failure before you waste 15 hours.

Usage:
    python dgx_healthcheck.py

What it checks:
  ✅ GPU / VRAM available
  ✅ CUDA + PyTorch working
  ✅ BF16 AMP working
  ✅ Dataset path exists and finds patients
  ✅ Can load a patient volume (no corrupt files)
  ✅ Patch extraction works
  ✅ DataLoader returns correct tensor shapes
  ✅ Model forward pass works (baseline + attention)
  ✅ Loss function works
  ✅ Backward pass + gradient update works
  ✅ Checkpoint save/load works
  ✅ Storage space sufficient
  ✅ RAM not at risk

Prints a GO / NO-GO summary at the end.
"""

import sys
import os
import time
import traceback
from pathlib import Path

# ── colour helpers ────────────────────────────────────────────────────────────
def green(t):  return f"\033[92m{t}\033[0m"
def red(t):    return f"\033[91m{t}\033[0m"
def yellow(t): return f"\033[93m{t}\033[0m"
def bold(t):   return f"\033[1m{t}\033[0m"

results = []  # list of (check_name, passed, message)

def check(name, fn):
    """Run a check function, catch exceptions, record result."""
    try:
        msg = fn()
        results.append((name, True, msg or "OK"))
        print(f"  {green('✅')} {name:50s} {green(msg or 'OK')}")
    except Exception as e:
        results.append((name, False, str(e)))
        print(f"  {red('❌')} {name:50s} {red(str(e)[:80])}")


# =============================================================================
print(bold("\n" + "="*70))
print(bold("  DGX Health Check — Brain Tumor Segmentation (BraTS 2023 GLI)"))
print(bold("  RCOEM B.Tech CSE-DS 2026-27  |  Run before any training!"))
print(bold("="*70))
# =============================================================================

# ── 1. Python & CUDA ──────────────────────────────────────────────────────────
print(bold("\n[1/8] Python & CUDA"))

def chk_python():
    v = sys.version_info
    assert v >= (3, 9), f"Need Python 3.9+, got {v.major}.{v.minor}"
    return f"Python {v.major}.{v.minor}.{v.micro}"

def chk_torch():
    import torch
    assert torch.__version__ >= "2.0", f"Need PyTorch 2.0+, got {torch.__version__}"
    return f"PyTorch {torch.__version__}"

def chk_cuda():
    import torch
    assert torch.cuda.is_available(), "CUDA not available — check nvidia-smi"
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    return f"{name}  |  {vram:.1f} GB VRAM"

def chk_vram():
    import torch
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    assert vram >= 10, f"Only {vram:.1f} GB VRAM — need at least 10 GB"
    return f"{vram:.1f} GB >= 10 GB minimum"

def chk_amp():
    import torch
    x = torch.randn(4, 4).cuda()
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        y = x @ x
    assert y.dtype == torch.bfloat16, "BF16 autocast did not produce bfloat16"
    return "BF16 autocast works"

check("Python version", chk_python)
check("PyTorch version", chk_torch)
check("CUDA available", chk_cuda)
check("VRAM sufficient (>=10 GB)", chk_vram)
check("BF16 AMP autocast", chk_amp)

# ── 2. Dependencies ───────────────────────────────────────────────────────────
print(bold("\n[2/8] Dependencies"))

def chk_dep(pkg):
    def _chk():
        mod = __import__(pkg.split(">=")[0].replace("-", "_"))
        ver = getattr(mod, "__version__", "unknown")
        return ver
    return _chk

for dep in ["nibabel", "monai", "sklearn", "tqdm", "tensorboard", "numpy", "scipy"]:
    check(f"import {dep}", chk_dep(dep))

# ── 3. Config & Paths ─────────────────────────────────────────────────────────
print(bold("\n[3/8] Config & Paths"))

sys.path.insert(0, str(Path(__file__).parent))

def chk_config():
    from src.config import (DATASET_PATH, PATCH_SIZE, BATCH_SIZE,
                             NUM_EPOCHS, USE_AMP, QUICK_TEST)
    return (f"patch={PATCH_SIZE}, bs={BATCH_SIZE}, "
            f"epochs={NUM_EPOCHS}, amp={USE_AMP}, quick_test={QUICK_TEST}")

def chk_dataset_path():
    from src.config import DATASET_PATH
    p = Path(DATASET_PATH)
    assert p.exists(), (
        f"Dataset path NOT FOUND: {DATASET_PATH}\n"
        f"  → Unzip the training data and update DATASET_PATH in src/config.py\n"
        f"  → Run: unzip ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip"
    )
    return f"Exists: {DATASET_PATH}"

def chk_patient_count():
    from src.config import DATASET_PATH
    from src.dataset import get_patient_folders
    folders = get_patient_folders(DATASET_PATH)
    assert len(folders) >= 100, f"Only {len(folders)} patients found — expected 1000+"
    return f"{len(folders)} patients found"

def chk_storage():
    import shutil
    stat = shutil.disk_usage("/")
    free_gb = stat.free / 1e9
    assert free_gb >= 50, f"Only {free_gb:.0f} GB free — need 50+ GB"
    return f"{free_gb:.0f} GB free disk space"

def chk_ram():
    import psutil
    ram_gb = psutil.virtual_memory().total / 1e9
    avail_gb = psutil.virtual_memory().available / 1e9
    assert avail_gb >= 8, f"Only {avail_gb:.1f} GB RAM available — need 8+ GB"
    return f"{avail_gb:.1f} GB available / {ram_gb:.0f} GB total"

check("src/config.py imports", chk_config)
check("DATASET_PATH exists", chk_dataset_path)
check("Patient count (>=100)", chk_patient_count)
check("Disk space (>=50 GB free)", chk_storage)
check("RAM available (>=8 GB)", chk_ram)

# ── 4. Data loading ───────────────────────────────────────────────────────────
print(bold("\n[4/8] Data Loading"))

def chk_load_patient():
    from src.config import DATASET_PATH
    from src.dataset import get_patient_folders, load_patient
    folders = get_patient_folders(DATASET_PATH)
    t0 = time.time()
    img, seg = load_patient(folders[0], has_seg=True)
    elapsed = time.time() - t0
    assert img.shape[0] == 4, f"Expected 4 modalities, got {img.shape[0]}"
    assert seg is not None, "No segmentation returned"
    return f"img={img.shape}  seg={seg.shape}  in {elapsed:.1f}s"

def chk_patch():
    from src.config import DATASET_PATH, PATCH_SIZE
    from src.dataset import get_patient_folders, load_patient, extract_random_patch
    folders = get_patient_folders(DATASET_PATH)
    img, seg = load_patient(folders[0])
    ip, sp = extract_random_patch(img, seg, PATCH_SIZE)
    assert ip.shape == (4,) + tuple(PATCH_SIZE), f"Wrong patch shape: {ip.shape}"
    return f"patch={ip.shape}"

def chk_dataloader():
    import torch
    from src.config import DATASET_PATH, PATCH_SIZE, BATCH_SIZE
    from src.dataset import get_patient_folders, BraTS2023Dataset
    from torch.utils.data import DataLoader
    folders = get_patient_folders(DATASET_PATH)[:4]  # only 4 patients for speed
    ds = BraTS2023Dataset(folders, patch_size=PATCH_SIZE, num_patches=1, augment=False)
    dl = DataLoader(ds, batch_size=1, num_workers=0)
    imgs, segs = next(iter(dl))
    assert imgs.shape[1] == 4, f"Expected 4 channels, got {imgs.shape[1]}"
    return f"batch={tuple(imgs.shape)}  seg={tuple(segs.shape)}"

check("Load 1 patient volume", chk_load_patient)
check("Extract 3D patch", chk_patch)
check("DataLoader batch shape", chk_dataloader)

# ── 5. Model forward pass ─────────────────────────────────────────────────────
print(bold("\n[5/8] Model Forward Pass"))

def chk_unet3d():
    import torch
    from src.config import PATCH_SIZE, BATCH_SIZE
    from src.models.unet3d import UNet3D
    model = UNet3D(in_channels=4, out_channels=4, init_features=32).cuda()
    x = torch.randn(1, 4, *PATCH_SIZE).cuda()
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        out = model(x)
    assert out.shape == (1, 4, *PATCH_SIZE), f"Wrong output shape: {out.shape}"
    params = sum(p.numel() for p in model.parameters()) / 1e6
    return f"out={tuple(out.shape)}  params={params:.1f}M"

def chk_attention_unet():
    import torch
    from src.config import PATCH_SIZE
    from src.models.attention_unet3d import AttentionUNet3D
    model = AttentionUNet3D(in_channels=4, out_channels=4, init_features=32).cuda()
    x = torch.randn(1, 4, *PATCH_SIZE).cuda()
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        out = model(x)
    assert out.shape == (1, 4, *PATCH_SIZE), f"Wrong output shape: {out.shape}"
    params = sum(p.numel() for p in model.parameters()) / 1e6
    return f"out={tuple(out.shape)}  params={params:.1f}M"

check("3D U-Net forward pass (BF16)", chk_unet3d)
check("Attention U-Net forward pass (BF16)", chk_attention_unet)

# ── 6. Loss + backward ────────────────────────────────────────────────────────
print(bold("\n[6/8] Loss + Backward Pass"))

def chk_loss_backward():
    import torch
    from src.config import PATCH_SIZE
    from src.models.attention_unet3d import AttentionUNet3D
    from src.losses import CombinedLoss

    model = AttentionUNet3D(in_channels=4, out_channels=4).cuda()
    criterion = CombinedLoss(num_classes=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    x = torch.randn(1, 4, *PATCH_SIZE).cuda()
    y = torch.randint(0, 4, (1, *PATCH_SIZE)).cuda()

    optimizer.zero_grad()
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        logits = model(x)
        loss, dl, fl = criterion(logits, y)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    assert loss.item() < 5.0, f"Loss too high: {loss.item():.2f} — possible NaN"
    return f"loss={loss.item():.4f}  dice={dl.item():.4f}  focal={fl.item():.4f}"

check("Loss + AMP backward pass", chk_loss_backward)

# ── 7. Checkpoint ─────────────────────────────────────────────────────────────
print(bold("\n[7/8] Checkpoint Save/Load"))

def chk_checkpoint():
    import torch
    from src.models.unet3d import UNet3D
    from src.utils import save_checkpoint, load_checkpoint

    model = UNet3D().cuda()
    opt   = torch.optim.Adam(model.parameters())
    save_checkpoint(model, opt, epoch=0, val_dice=0.5,
                    model_name="healthcheck_test", is_best=True)
    start_ep, best_dice = load_checkpoint(model, opt, "healthcheck_test")
    # Cleanup
    import glob
    for f in glob.glob("./checkpoints/healthcheck_test*.pth"):
        os.remove(f)
    return f"Save + load OK  (resumed epoch={start_ep})"

check("Checkpoint save + load", chk_checkpoint)

# ── 8. VRAM stress test ───────────────────────────────────────────────────────
print(bold("\n[8/8] VRAM Stress Test (batch=4, patch=128)"))

def chk_vram_stress():
    import torch
    from src.config import PATCH_SIZE, BATCH_SIZE
    from src.models.attention_unet3d import AttentionUNet3D
    from src.losses import CombinedLoss

    model = AttentionUNet3D(in_channels=4, out_channels=4).cuda()
    criterion = CombinedLoss()
    scaler = torch.cuda.amp.GradScaler()
    opt = torch.optim.Adam(model.parameters())

    x = torch.randn(BATCH_SIZE, 4, *PATCH_SIZE).cuda()
    y = torch.randint(0, 4, (BATCH_SIZE, *PATCH_SIZE)).cuda()

    opt.zero_grad()
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        logits = model(x)
        loss, _, _ = criterion(logits, y)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()

    used_gb  = torch.cuda.max_memory_allocated() / 1e9
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    assert used_gb < total_gb * 0.95, (
        f"VRAM nearly full: {used_gb:.1f}/{total_gb:.1f} GB — reduce BATCH_SIZE or PATCH_SIZE"
    )
    return f"Peak VRAM: {used_gb:.1f} / {total_gb:.1f} GB  ({100*used_gb/total_gb:.0f}% used)"

check(f"Full batch forward+backward", chk_vram_stress)

# =============================================================================
# FINAL SUMMARY
# =============================================================================
passed = [r for r in results if r[1]]
failed = [r for r in results if not r[1]]

print(bold("\n" + "="*70))
print(bold("  HEALTH CHECK SUMMARY"))
print(bold("="*70))
print(f"  {green('Passed')}: {len(passed)} / {len(results)}")
if failed:
    print(f"  {red('Failed')}: {len(failed)} / {len(results)}")
    print(bold("\n  ❌ FAILED CHECKS — FIX BEFORE TRAINING:"))
    for name, _, msg in failed:
        print(f"    • {name}")
        print(f"      → {msg}")

print()
if not failed:
    print(bold(green("  🚀 ALL CHECKS PASSED — GO AHEAD AND START TRAINING!")))
    print(f"""
  Next steps:
    1. Set QUICK_TEST = True in src/config.py (already set)
    2. Open notebooks/03_train_baseline.ipynb and run all cells
    3. Confirm Dice > 0.30 at epoch 5
    4. Set QUICK_TEST = False and relaunch for full training
""")
else:
    print(bold(red("  ⛔ DO NOT START TRAINING YET — fix the issues above first.")))

print("="*70 + "\n")
sys.exit(0 if not failed else 1)
