import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.utils import _get_dim_steps, sliding_window_inference

H, W, D = 240, 240, 155
ph, pw, pd = 96, 96, 96
oh, ow, od = 48, 48, 48  # 50% overlap

sh = ph - oh  # stride = 48
sw = pw - ow  # stride = 48
sd = pd - od  # stride = 48

h_steps = _get_dim_steps(H, ph, sh)
w_steps = _get_dim_steps(W, pw, sw)
d_steps = _get_dim_steps(D, pd, sd)

print(f"Volume Shape: ({H}, {W}, {D}) | Patch Size: ({ph}, {pw}, {pd}) | Stride: ({sh}, {sw}, {sd})")
print(f"H Steps (Y-axis starts): {h_steps} -> Slices: {[(y, y+ph) for y in h_steps]}")
print(f"W Steps (X-axis starts): {w_steps} -> Slices: {[(x, x+pw) for x in w_steps]}")
print(f"D Steps (Z-axis starts): {d_steps} -> Slices: {[(z, z+pd) for z in d_steps]}")
print(f"Total Patches per Volume: {len(h_steps)} x {len(w_steps)} x {len(d_steps)} = {len(h_steps) * len(w_steps) * len(d_steps)}")

# Track voxel coverage matrix
coverage = np.zeros((H, W, D), dtype=np.int32)
visited_patches = []

for zi in d_steps:
    for yi in h_steps:
        for xi in w_steps:
            ye = yi + ph
            xe = xi + pw
            ze = zi + pd
            visited_patches.append(((yi, ye), (xi, xe), (zi, ze)))
            coverage[yi:ye, xi:xe, zi:ze] += 1

print("\n--- Spatial Boundary Verification ---")
print(f"Voxel (0, 0, 0) Coverage Count       : {coverage[0, 0, 0]}")
print(f"Voxel (239, 239, 154) Coverage Count : {coverage[239, 239, 154]}")
print(f"Minimum Voxel Coverage Across Volume : {coverage.min()} (must be >= 1)")
print(f"Maximum Voxel Coverage Across Volume : {coverage.max()}")
print(f"Uncovered Voxels (coverage == 0)     : {(coverage == 0).sum()} / {coverage.size}")
assert (coverage >= 1).all(), "CRITICAL: Some voxels were not covered by sliding window!"
print("STATUS: 100% VOXEL COVERAGE VERIFIED FROM (0,0,0) TO (239,239,154)")

# Test with mock model
class MockSegmenter(nn.Module):
    def forward(self, x):
        # x: (B, C, ph, pw, pd)
        B, _, h, w, d = x.shape
        out = torch.zeros((B, 4, h, w, d), dtype=torch.float32)
        out[:, 3, :, :, :] = 10.0  # predict ET (class 3) everywhere
        return out

mock_model = MockSegmenter()
dummy_volume = torch.randn(4, H, W, D, dtype=torch.float32)
pred_seg = sliding_window_inference(mock_model, dummy_volume, patch_size=(ph, pw, pd), overlap=(oh, ow, od))

print(f"\nPrediction Output Shape: {pred_seg.shape}")
print(f"Unique Predicted Classes: {np.unique(pred_seg)}")
assert pred_seg.shape == (H, W, D)
assert (pred_seg == 3).all(), "Prediction failed to cover entire volume!"
print("STATUS: Full sliding_window_inference() end-to-end execution PASSED.")
