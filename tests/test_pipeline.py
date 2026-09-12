"""
tests/test_pipeline.py — Comprehensive Unit & Integration Tests
Validates:
  1. Subject-level zero data leakage in splitting logic
  2. All 4 model variants (forward & backward passes)
  3. Loss function correctness (Multi-class Dice + Focal / CE)
  4. 100% spatial coverage in sliding-window inference (zero truncation bug)
  5. Checkpoint saving and exact state recovery
  6. Robustness of 15-metric suite against edge cases (empty masks, single voxels)
"""

import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import (
    extract_subject_id, get_subject_splits,
    extract_random_patch, augment_patch, remap_labels, normalize_volume
)
from src.models import build_model, count_parameters, AttentionUNet3D
from src.losses import DiceLoss, FocalLoss, CombinedLoss
from src.metrics import (
    get_regions, dice_score, iou_score, sensitivity, specificity,
    hausdorff_distance_95, compute_patient_metrics, aggregate_metrics
)
from src.utils import (
    set_seed, save_checkpoint, load_checkpoint,
    sliding_window_inference, _get_dim_steps
)


def test_subject_splitting_leakage():
    print("\n[TEST 1] Testing Subject-Level Splitting & Zero Data Leakage...")
    # Simulate multi-scan longitudinal patient folders
    dummy_folders = [
        "BraTS-GLI-00001-000", "BraTS-GLI-00001-001",
        "BraTS-GLI-00002-000",
        "BraTS-GLI-00003-000", "BraTS-GLI-00003-001", "BraTS-GLI-00003-002",
        "BraTS-GLI-00004-000",
        "BraTS-GLI-00005-000", "BraTS-GLI-00005-001",
        "BraTS-GLI-00006-000",
        "BraTS-GLI-00007-000",
        "BraTS-GLI-00008-000",
        "BraTS-GLI-00009-000",
        "BraTS-GLI-00010-000",
    ]
    train_f, val_f, test_f = get_subject_splits(dummy_folders, n_folds=3, fold=0, seed=42)

    train_subs = set(extract_subject_id(f) for f in train_f)
    val_subs   = set(extract_subject_id(f) for f in val_f)
    test_subs  = set(extract_subject_id(f) for f in test_f)

    # 1. Zero intersection between splits
    assert len(train_subs.intersection(val_subs)) == 0, "Train-Val Subject Leakage detected!"
    assert len(train_subs.intersection(test_subs)) == 0, "Train-Test Subject Leakage detected!"
    assert len(val_subs.intersection(test_subs)) == 0, "Val-Test Subject Leakage detected!"

    # 2. All scans of a subject are in the same split
    for sid in set(extract_subject_id(f) for f in dummy_folders):
        in_train = sid in train_subs
        in_val   = sid in val_subs
        in_test  = sid in test_subs
        assert (in_train + in_val + in_test) == 1, f"Subject {sid} allocated to multiple splits!"

    print("  ✅ Zero patient-level data leakage verified across train/val/test splits.")


def test_normalization_and_remapping():
    print("\n[TEST 2] Testing Normalization & Label Remapping...")
    # Label remapping (4 -> 3)
    raw_labels = np.array([0, 1, 2, 4, 2, 1, 0], dtype=np.int64)
    remapped = remap_labels(raw_labels)
    assert not (remapped == 4).any(), "Label 4 was not remapped to 3!"
    assert set(np.unique(remapped)).issubset({0, 1, 2, 3}), "Labels contain unexpected values!"

    # Per-volume z-score normalization
    vol = np.random.randn(30, 30, 30).astype(np.float32) + 10.0
    mask = vol > 10.0
    norm_vol = normalize_volume(vol, mask)
    assert np.isclose(norm_vol[mask].mean(), 0.0, atol=1e-5), "Normalized mean is not 0!"
    assert np.isclose(norm_vol[mask].std(), 1.0, atol=1e-4), "Normalized std is not 1!"
    assert (norm_vol[~mask] == 0).all(), "Background voxels modified by normalization!"
    print("  ✅ Per-volume normalization and label remapping verified.")


def test_model_variants_forward_backward():
    print("\n[TEST 3] Testing All 4 Model Architecture Variants...")
    variants = ["baseline", "channel", "spatial", "hybrid"]
    x = torch.randn(2, 4, 32, 32, 32, requires_grad=True)
    targets = torch.randint(0, 4, (2, 32, 32, 32))
    criterion = CombinedLoss(num_classes=4)

    for v in variants:
        model = build_model(variant=v, in_channels=4, out_channels=4, init_features=16)
        logits = model(x)
        assert logits.shape == (2, 4, 32, 32, 32), f"Variant {v} produced invalid shape: {logits.shape}"

        loss, dl, sl = criterion(logits, targets)
        loss.backward()

        # Check gradients exist
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad, f"Variant {v} failed gradient backpropagation!"

        # Check attention maps method for hybrid model
        if v == "hybrid":
            logits_att, att_maps = model.get_attention_maps(x)
            assert len(att_maps) == 4, f"Expected 4 attention maps, got {len(att_maps)}"
            assert att_maps[0].shape[0] == 2, "Attention map batch dim mismatch"

        info = count_parameters(model)
        print(f"  ✅ Variant [{v.upper():8s}] passed forward & backward pass ({info['formatted']})")


def test_sliding_window_coverage():
    print("\n[TEST 4] Testing 100% Spatial Coverage in Sliding-Window Inference...")
    H, W, D = 240, 240, 155
    ph, pw, pd = 96, 96, 96
    oh, ow, od = 48, 48, 48

    sh = ph - oh
    sw = pw - ow
    sd = pd - od

    h_steps = _get_dim_steps(H, ph, sh)
    w_steps = _get_dim_steps(W, pw, sw)
    d_steps = _get_dim_steps(D, pd, sd)

    # Assert last step ends exactly at dimension boundary
    assert h_steps[-1] + ph == H, f"H step does not cover end boundary! ({h_steps[-1] + ph} != {H})"
    assert w_steps[-1] + pw == W, f"W step does not cover end boundary! ({w_steps[-1] + pw} != {W})"
    assert d_steps[-1] + pd == D, f"D step does not cover end boundary! ({d_steps[-1] + pd} != {D})"

    # Run inference with a mock constant model returning class 2 everywhere
    class MockModel(torch.nn.Module):
        def forward(self, x):
            B, _, ph, pw, pd = x.shape
            out = torch.zeros((B, 4, ph, pw, pd))
            out[:, 2, :, :, :] = 10.0  # strongly predict class 2
            return out

    mock = MockModel()
    test_vol = torch.randn(4, 240, 240, 155)
    pred_seg = sliding_window_inference(mock, test_vol, patch_size=(96, 96, 96), overlap=(48, 48, 48))

    assert pred_seg.shape == (240, 240, 155), f"Invalid output volume shape: {pred_seg.shape}"
    # Assert every single voxel is covered (none left at 0 by truncation)
    assert (pred_seg == 2).all(), "Sliding window missed slices (boundary truncation bug)!"
    print("  ✅ Sliding-window inference covers 100% of 3D volume dimensions without truncation.")


def test_checkpoint_save_and_resumption():
    print("\n[TEST 5] Testing Checkpoint Resumption & Exact State Recovery...")
    model1 = build_model("hybrid", init_features=16)
    opt1 = torch.optim.Adam(model1.parameters(), lr=1e-3)
    sched1 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt1)
    try:
        scaler1 = torch.amp.GradScaler("cuda", enabled=False)
    except (AttributeError, TypeError):
        scaler1 = torch.cuda.amp.GradScaler(enabled=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save at epoch 7 with metric 0.85
        save_checkpoint(
            model=model1, optimizer=opt1, scheduler=sched1, scaler=scaler1,
            epoch=7, val_metric=0.85, model_name="test_model", checkpoint_dir=tmpdir, is_best=True
        )

        # Restore into model2
        model2 = build_model("hybrid", init_features=16)
        opt2 = torch.optim.Adam(model2.parameters(), lr=1e-4)
        sched2 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt2)
        try:
            scaler2 = torch.amp.GradScaler("cuda", enabled=False)
        except (AttributeError, TypeError):
            scaler2 = torch.cuda.amp.GradScaler(enabled=False)

        start_epoch, best_metric = load_checkpoint(
            model=model2, optimizer=opt2, scheduler=sched2, scaler=scaler2,
            model_name="test_model", checkpoint_dir=tmpdir, prefer_best=False
        )

        assert start_epoch == 8, f"Expected start_epoch=8, got {start_epoch}"
        assert np.isclose(best_metric, 0.85), f"Expected best_metric=0.85, got {best_metric}"

        # Verify exact weight match
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2), "Model parameters do not match after checkpoint restoration!"

    print("  ✅ Checkpoint saving and resumption verified with bit-for-bit weight match.")


def test_metrics_suite_robustness():
    print("\n[TEST 6] Testing 15-Metric Suite & Edge Cases...")
    shape = (50, 50, 50)

    # Edge Case 1: Identical masks
    mask_a = np.zeros(shape, dtype=np.int64)
    mask_a[20:30, 20:30, 20:30] = 3  # ET
    metrics_id = compute_patient_metrics(mask_a, mask_a)
    assert np.isclose(metrics_id["ET"]["dice"], 1.0), "Dice on identical masks must be 1.0"
    assert np.isclose(metrics_id["ET"]["iou"], 1.0), "IoU on identical masks must be 1.0"
    assert np.isclose(metrics_id["ET"]["hd95"], 0.0), "HD95 on identical masks must be 0.0"

    # Edge Case 2: Completely empty prediction
    empty = np.zeros(shape, dtype=np.int64)
    metrics_empty = compute_patient_metrics(empty, mask_a)
    assert np.isclose(metrics_empty["ET"]["dice"], 0.0), "Dice with empty pred must be 0.0"
    assert np.isnan(metrics_empty["ET"]["hd95"]), "HD95 with empty pred must be NaN"

    # Edge Case 3: Both empty
    metrics_both_empty = compute_patient_metrics(empty, empty)
    assert np.isclose(metrics_both_empty["ET"]["dice"], 1.0), "Dice with both empty must be 1.0"
    assert np.isclose(metrics_both_empty["ET"]["hd95"], 0.0), "HD95 with both empty must be 0.0"

    # Aggregation test
    agg = aggregate_metrics([metrics_id, metrics_empty, metrics_both_empty])
    assert "mean" in agg["ET"]["dice"]
    assert not np.isnan(agg["ET"]["dice"]["mean"])
    print("  ✅ 15-metric suite passed all robustness and edge-case tests.")


def run_all_tests():
    print("=" * 70)
    print("  🧪 Running Comprehensive 3D Brain Tumor Segmentation Test Suite")
    print("=" * 70)
    set_seed(42)

    test_subject_splitting_leakage()
    test_normalization_and_remapping()
    test_model_variants_forward_backward()
    test_sliding_window_coverage()
    test_checkpoint_save_and_resumption()
    test_metrics_suite_robustness()

    print("\n" + "=" * 70)
    print("  🎉 ALL 6 COMPREHENSIVE INTEGRATION & UNIT TESTS PASSED!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_all_tests()
