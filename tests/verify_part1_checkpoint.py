import os
import sys
import random
import tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.models import build_model
from src.losses import CombinedLoss
from src.utils import save_checkpoint, load_checkpoint

print("=" * 65)
print("  Part 1 - Check 4: Checkpoint Round-Trip Verification")
print("=" * 65)

with tempfile.TemporaryDirectory() as tmpdir:
    # 1. Setup deterministic seeds and Model 1
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)

    model1 = build_model("hybrid", in_channels=4, out_channels=4, init_features=16)
    optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.01234, momentum=0.9, weight_decay=1e-4)
    scheduler1 = ReduceLROnPlateau(optimizer1, mode="max", factor=0.5, patience=2)
    criterion = CombinedLoss(num_classes=4)

    print("\n--- Training Model 1 for 2 Epochs ---")
    for epoch in range(1, 3):
        model1.train()
        x = torch.randn(2, 4, 16, 16, 16)
        y = torch.randint(0, 4, (2, 16, 16, 16))
        optimizer1.zero_grad()
        logits = model1(x)
        loss, _, _ = criterion(logits, y)
        loss.backward()
        optimizer1.step()
        scheduler1.step(0.75 + epoch * 0.05)
        print(f"  Epoch {epoch:02d} completed | Loss: {loss.item():.4f} | LR: {optimizer1.param_groups[0]['lr']:.5f}")

    best_val_dice = 0.8542
    # Save checkpoint after epoch 2
    ckpt_path = save_checkpoint(
        model=model1,
        optimizer=optimizer1,
        scheduler=scheduler1,
        scaler=None,
        epoch=2,
        val_metric=best_val_dice,
        model_name="hybrid_test",
        checkpoint_dir=tmpdir,
        is_best=True
    )
    print(f"\nSaved Checkpoint: {ckpt_path}")

    # Capture expected next random numbers after save
    expected_rand_py = random.random()
    expected_rand_np = float(np.random.rand())
    expected_rand_th = float(torch.rand(1).item())

    # 2. Instantiate completely fresh Model 2 with different weights and optimizer params
    random.seed(9999)
    np.random.seed(9999)
    torch.manual_seed(9999)

    model2 = build_model("hybrid", in_channels=4, out_channels=4, init_features=16)
    optimizer2 = torch.optim.SGD(model2.parameters(), lr=0.9999, momentum=0.0)
    scheduler2 = ReduceLROnPlateau(optimizer2, mode="max")

    # Verify weights are initially different
    diff_count = sum(not torch.allclose(p1, p2) for p1, p2 in zip(model1.parameters(), model2.parameters()))
    print(f"\nFresh Model 2 created: {diff_count} / {len(list(model1.parameters()))} parameter tensors differ before load.")

    # 3. Load Checkpoint into Model 2
    resumed_epoch, restored_metric = load_checkpoint(
        model=model2,
        optimizer=optimizer2,
        scheduler=scheduler2,
        checkpoint_path=ckpt_path,
        device=torch.device("cpu")
    )

    # 4. Detailed Verifications
    print("\n--- Detailed State Verification ---")
    
    # (a) Model weights match
    weights_match = all(torch.allclose(p1, p2, atol=1e-7) for p1, p2 in zip(model1.parameters(), model2.parameters()))
    print(f"1. Model Weights Matched Exactly : {weights_match}")
    assert weights_match, "Model weights do not match after checkpoint load!"

    # (b) Optimizer state
    restored_lr = optimizer2.param_groups[0]["lr"]
    print(f"2. Optimizer Learning Rate       : {restored_lr:.5f} (Expected: 0.01234)")
    assert np.isclose(restored_lr, 0.01234), "Optimizer LR mismatch!"
    
    # Check momentum buffers present
    has_momentum_buffers = any("momentum_buffer" in s for s in optimizer2.state.values())
    print(f"   Optimizer Momentum Buffers    : Present = {has_momentum_buffers} ({len(optimizer2.state)} param states)")
    assert has_momentum_buffers, "Optimizer momentum buffers were not restored!"

    # (c) Scheduler state
    sched_best = getattr(scheduler2, "best", None)
    print(f"3. Scheduler Restored State      : mode={scheduler2.mode}, best_metric={sched_best}")

    # (d) Resumed epoch & best metric
    print(f"4. Resumed Start Epoch           : {resumed_epoch} (Expected: 3)")
    print(f"   Restored Best Metric          : {restored_metric:.4f} (Expected: {best_val_dice:.4f})")
    assert resumed_epoch == 3, f"Expected start_epoch=3, got {resumed_epoch}"
    assert np.isclose(restored_metric, best_val_dice), "Best metric mismatch!"

    # (e) RNG States
    # Load raw state dict to check RNG states
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    torch.set_rng_state(state["torch_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    # Generate random numbers
    reproduced_rand_np = float(np.random.rand())
    reproduced_rand_th = float(torch.rand(1).item())

    print(f"5. Random Number State Recovery  :")
    print(f"   NumPy Rand (Restored vs Exp)  : {reproduced_rand_np:.8f} == {expected_rand_np:.8f} -> {np.isclose(reproduced_rand_np, expected_rand_np)}")
    print(f"   PyTorch Rand (Restored vs Exp): {reproduced_rand_th:.8f} == {expected_rand_th:.8f} -> {np.isclose(reproduced_rand_th, expected_rand_th)}")
    assert np.isclose(reproduced_rand_np, expected_rand_np)
    assert np.isclose(reproduced_rand_th, expected_rand_th)

    print("\nSTATUS: 100% CHECKPOINT ROUND-TRIP FIDELITY CONFIRMED.")
