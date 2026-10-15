"""
dataset_analysis.py - BraTS 2023 GLI Dataset Class Imbalance & Distribution Analysis
Run:  python dataset_analysis.py
Output: dataset_analysis_report.txt + dataset_analysis_plots.png
"""

import os
import sys
import time
import numpy as np
import nibabel as nib
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CONFIG
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DATASET_ROOT = Path(r"C:\Users\Rishil\Downloads\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData")
OUTPUT_DIR   = Path(r"C:\Users\Rishil\Documents\Brain_Tumor_Segmentation\results")
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_PATIENTS = None   # None = all 1251; set to e.g. 100 for a quick run

# BraTS 2023 label definitions
LABEL_NAMES = {0: "Background", 1: "NCR (Necrotic Core)", 2: "ED (Edema)", 3: "ET (Enhancing Tumor)"}
REGION_NAMES = {"WT": "Whole Tumor (1+2+3)", "TC": "Tumor Core (1+3)", "ET": "Enhancing Tumor (3)"}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Discover patients
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
patient_dirs = sorted([
    d for d in DATASET_ROOT.iterdir()
    if d.is_dir() and d.name.startswith("BraTS-GLI-")
])
if MAX_PATIENTS:
    patient_dirs = patient_dirs[:MAX_PATIENTS]

N = len(patient_dirs)
print(f"\n{'='*65}")
print(f"  BraTS 2023 GLI  â€”  Dataset Analysis")
print(f"  Patients found : {N}")
print(f"  Dataset root   : {DATASET_ROOT}")
print(f"{'='*65}\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Per-patient accumulators
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
label_voxel_counts   = defaultdict(list)   # {label_id: [count_per_patient]}
region_voxel_counts  = defaultdict(list)   # {region: [count_per_patient]}
total_brain_voxels   = []
patients_with_et     = 0
patients_missing_seg = 0

start = time.time()

for i, pdir in enumerate(patient_dirs):
    seg_file = pdir / f"{pdir.name}-seg.nii.gz"
    flair_file = pdir / f"{pdir.name}-t2f.nii.gz"

    if not seg_file.exists():
        patients_missing_seg += 1
        continue

    seg  = nib.load(str(seg_file)).get_fdata().astype(np.int8).ravel()
    flair = nib.load(str(flair_file)).get_fdata().astype(np.float32)

    brain_voxels = int((flair > 0).sum())
    total_brain_voxels.append(brain_voxels)

    for lbl in [0, 1, 2, 3]:
        label_voxel_counts[lbl].append(int((seg == lbl).sum()))

    wt = int(((seg == 1) | (seg == 2) | (seg == 3)).sum())
    tc = int(((seg == 1) | (seg == 3)).sum())
    et = int((seg == 3).sum())
    region_voxel_counts["WT"].append(wt)
    region_voxel_counts["TC"].append(tc)
    region_voxel_counts["ET"].append(et)

    if et > 0:
        patients_with_et += 1

    # Progress
    if (i + 1) % 50 == 0 or (i + 1) == N:
        elapsed = time.time() - start
        eta     = elapsed / (i + 1) * (N - i - 1)
        print(f"  [{i+1:4d}/{N}]  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

elapsed_total = time.time() - start
print(f"\nScan complete in {elapsed_total:.1f}s\n")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Compute aggregate statistics
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
TOTAL_VOXELS_PER_VOL = 240 * 240 * 155   # = 8,928,000

def stats(arr):
    a = np.array(arr)
    return dict(
        mean=float(np.mean(a)), std=float(np.std(a)),
        median=float(np.median(a)), min=float(np.min(a)),
        max=float(np.max(a)), p25=float(np.percentile(a, 25)),
        p75=float(np.percentile(a, 75)), p5=float(np.percentile(a, 5)),
        p95=float(np.percentile(a, 95)),
    )

label_stats  = {lbl: stats(label_voxel_counts[lbl])  for lbl in [0, 1, 2, 3]}
region_stats = {r:   stats(region_voxel_counts[r])   for r in ["WT", "TC", "ET"]}
brain_stats  = stats(total_brain_voxels)

# Global voxel totals (across all patients)
global_label_total = {lbl: sum(label_voxel_counts[lbl]) for lbl in [0, 1, 2, 3]}
global_total       = sum(global_label_total.values())
global_label_pct   = {lbl: 100.0 * global_label_total[lbl] / global_total for lbl in [0, 1, 2, 3]}

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Build text report
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
lines = []
def p(s=""): lines.append(s); print(s)

p("=" * 65)
p("  BraTS 2023 GLI  â€”  Full Dataset Analysis Report")
p(f"  Generated from {N} patients  |  {elapsed_total:.1f}s runtime")
p("=" * 65)

# â”€â”€ 1. Dataset overview â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
p("\nâ”€â”€ 1. DATASET OVERVIEW â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
p(f"  Total patients analysed          : {N}")
p(f"  Patients missing seg file        : {patients_missing_seg}")
p(f"  Patients WITH Enhancing Tumor    : {patients_with_et} / {N} ({100*patients_with_et/N:.1f}%)")
p(f"  Patients WITHOUT Enhancing Tumor : {N - patients_with_et} / {N} ({100*(N-patients_with_et)/N:.1f}%)")
p(f"  Volume dimensions                : 240 Ã— 240 Ã— 155 voxels")
p(f"  Voxels per volume                : {TOTAL_VOXELS_PER_VOL:,}")
p(f"  Mean brain voxels per patient    : {brain_stats['mean']:,.0f} ({100*brain_stats['mean']/TOTAL_VOXELS_PER_VOL:.1f}% of volume)")

# â”€â”€ 2. Global class imbalance â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
p("\nâ”€â”€ 2. GLOBAL CLASS IMBALANCE (all voxels, all patients) â”€â”€â”€â”€â”€â”€â”€â”€")
p(f"  {'Label':<30}  {'Total Voxels':>15}  {'% of All Voxels':>16}  {'Imbalance Ratio':>16}")
p(f"  {'-'*75}")
for lbl in [0, 1, 2, 3]:
    ratio = global_label_total[0] / max(global_label_total[lbl], 1)
    p(f"  {LABEL_NAMES[lbl]:<30}  {global_label_total[lbl]:>15,}  {global_label_pct[lbl]:>15.4f}%  {ratio:>15.1f}Ã—")

# â”€â”€ 3. Per-patient label statistics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
p("\nâ”€â”€ 3. PER-PATIENT VOXEL COUNTS (per raw label) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
p(f"  {'Label':<30}  {'Mean':>10}  {'Std':>10}  {'Median':>10}  {'Min':>8}  {'Max':>10}  {'P5':>8}  {'P95':>10}")
p(f"  {'-'*95}")
for lbl in [0, 1, 2, 3]:
    s = label_stats[lbl]
    p(f"  {LABEL_NAMES[lbl]:<30}  {s['mean']:>10,.0f}  {s['std']:>10,.0f}  {s['median']:>10,.0f}  {s['min']:>8,.0f}  {s['max']:>10,.0f}  {s['p5']:>8,.0f}  {s['p95']:>10,.0f}")

# â”€â”€ 4. BraTS sub-region statistics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
p("\nâ”€â”€ 4. BRATS SUB-REGION STATISTICS (WT / TC / ET) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
p(f"  {'Region':<30}  {'Mean':>10}  {'Std':>10}  {'Median':>10}  {'Min':>8}  {'Max':>10}  {'P5':>8}  {'P95':>10}")
p(f"  {'-'*95}")
for r in ["WT", "TC", "ET"]:
    s = region_stats[r]
    p(f"  {REGION_NAMES[r]:<30}  {s['mean']:>10,.0f}  {s['std']:>10,.0f}  {s['median']:>10,.0f}  {s['min']:>8,.0f}  {s['max']:>10,.0f}  {s['p5']:>8,.0f}  {s['p95']:>10,.0f}")

# â”€â”€ 5. Class imbalance ratios â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
p("\nâ”€â”€ 5. WHAT THE IMBALANCE MEANS FOR TRAINING â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
bg_mean  = label_stats[0]['mean']
ncr_mean = label_stats[1]['mean']
ed_mean  = label_stats[2]['mean']
et_mean  = label_stats[3]['mean']
p(f"  Mean Background voxels     : {bg_mean:>10,.0f}  (100%)")
p(f"  Mean NCR voxels            : {ncr_mean:>10,.0f}  ({100*ncr_mean/bg_mean:.3f}% of BG)  â†’  BG:NCR = {bg_mean/max(ncr_mean,1):.0f}:1")
p(f"  Mean ED voxels             : {ed_mean:>10,.0f}  ({100*ed_mean/bg_mean:.3f}% of BG)   â†’  BG:ED  = {bg_mean/max(ed_mean,1):.0f}:1")
p(f"  Mean ET voxels             : {et_mean:>10,.0f}  ({100*et_mean/bg_mean:.4f}% of BG) â†’  BG:ET  = {bg_mean/max(et_mean,1):.0f}:1")
p()
p("  â†’ This extreme imbalance (BG dominates by 100-1000x) is WHY:")
p("    1. We use Dice Loss (overlap-based, not count-based)")
p("    2. We use Focal Loss (downweights easy background voxels)")
p("    3. We use foreground-biased patch sampling (67% tumor patches)")
p("    4. We ignore background in Dice metric calculation")

p("\n" + "=" * 65)
p("  Report saved to: dataset_analysis_report.txt")
p("=" * 65)

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Save text report
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
report_path = OUTPUT_DIR / "dataset_analysis_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"\nReport saved --> {report_path}")

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Plots
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    BG_COL = "#0d1117"
    FG_COL = "#c9d1d9"
    COLORS  = ["#8b949e", "#ff6464", "#64b4ff", "#64ff96"]
    R_COLORS = ["#3fb950", "#f0883e", "#58a6ff"]

    fig = plt.figure(figsize=(22, 18), facecolor=BG_COL)
    fig.suptitle(
        f"BraTS 2023 GLI â€” Dataset Class Imbalance & Distribution Analysis\n"
        f"{N} Patients | 240Ã—240Ã—155 voxels | 4 MRI modalities",
        color="#58a6ff", fontsize=14, fontweight="bold", y=0.99,
    )

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35,
                           top=0.93, bottom=0.06, left=0.07, right=0.97)

    def ax_style(ax, title):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors=FG_COL, labelsize=9)
        ax.title.set_color("#58a6ff")
        ax.title.set_fontsize(10)
        ax.title.set_fontweight("bold")
        ax.set_title(title)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.xaxis.label.set_color(FG_COL)
        ax.yaxis.label.set_color(FG_COL)

    # â”€â”€ Plot 1: Global label distribution (pie chart) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ax1 = fig.add_subplot(gs[0, 0])
    totals = [global_label_total[l] for l in [0, 1, 2, 3]]
    labels = ["BG", "NCR", "ED", "ET"]
    explode = [0, 0.07, 0.07, 0.12]
    wedges, texts, autotexts = ax1.pie(
        totals, labels=labels, colors=COLORS, autopct="%1.3f%%",
        startangle=90, explode=explode,
        textprops={"color": FG_COL, "fontsize": 9},
        pctdistance=0.75,
    )
    for at in autotexts:
        at.set_fontsize(8)
        at.set_color("#f0f6fc")
    ax1.set_facecolor(BG_COL)
    ax_style(ax1, "Global Label Distribution\n(all voxels, all patients)")

    # â”€â”€ Plot 2: Global label distribution (bar) â€” log scale â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ax2 = fig.add_subplot(gs[0, 1])
    bar_vals = [global_label_total[l] for l in [0, 1, 2, 3]]
    bars = ax2.bar(labels, bar_vals, color=COLORS, edgecolor="#30363d", width=0.6)
    ax2.set_yscale("log")
    ax2.set_ylabel("Total Voxels (log scale)", color=FG_COL)
    for bar, val in zip(bars, bar_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.2,
                 f"{val/1e9:.2f}B", ha="center", va="bottom",
                 color=FG_COL, fontsize=8)
    ax_style(ax2, "Class Volume (Log Scale)\nBackground vs Tumor Labels")

    # â”€â”€ Plot 3: BraTS regions bar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ax3 = fig.add_subplot(gs[0, 2])
    r_means = [region_stats[r]["mean"] for r in ["WT", "TC", "ET"]]
    r_stds  = [region_stats[r]["std"]  for r in ["WT", "TC", "ET"]]
    r_labels = ["WT\n(Whole)", "TC\n(Core)", "ET\n(Enhancing)"]
    bars3 = ax3.bar(r_labels, r_means, yerr=r_stds, capsize=5,
                    color=R_COLORS, edgecolor="#30363d", width=0.55,
                    error_kw={"ecolor": "#8b949e", "lw": 1.5})
    ax3.set_ylabel("Mean Voxel Count Â± SD", color=FG_COL)
    for bar, val in zip(bars3, r_means):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.02,
                 f"{val:,.0f}", ha="center", va="bottom",
                 color=FG_COL, fontsize=8)
    ax_style(ax3, "Mean Voxel Count per BraTS Region\n(Mean Â± 1 SD across patients)")

    # â”€â”€ Plot 4: NCR distribution histogram â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(label_voxel_counts[1], bins=50, color=COLORS[1], edgecolor="none", alpha=0.85)
    ax4.axvline(label_stats[1]["median"], color="#f0f6fc", lw=1.5, linestyle="--", label=f"Median={label_stats[1]['median']:,.0f}")
    ax4.set_xlabel("NCR Voxels per Patient", color=FG_COL)
    ax4.set_ylabel("No. of Patients", color=FG_COL)
    ax4.legend(fontsize=8, facecolor="#161b22", labelcolor=FG_COL, edgecolor="#30363d")
    ax_style(ax4, "NCR (Necrotic Core) â€” Voxel Count Distribution")

    # â”€â”€ Plot 5: ED distribution histogram â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(label_voxel_counts[2], bins=50, color=COLORS[2], edgecolor="none", alpha=0.85)
    ax5.axvline(label_stats[2]["median"], color="#f0f6fc", lw=1.5, linestyle="--", label=f"Median={label_stats[2]['median']:,.0f}")
    ax5.set_xlabel("ED Voxels per Patient", color=FG_COL)
    ax5.set_ylabel("No. of Patients", color=FG_COL)
    ax5.legend(fontsize=8, facecolor="#161b22", labelcolor=FG_COL, edgecolor="#30363d")
    ax_style(ax5, "ED (Peritumoral Edema) â€” Voxel Count Distribution")

    # â”€â”€ Plot 6: ET distribution histogram â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.hist(label_voxel_counts[3], bins=50, color=COLORS[3], edgecolor="none", alpha=0.85)
    ax6.axvline(label_stats[3]["median"], color="#f0f6fc", lw=1.5, linestyle="--", label=f"Median={label_stats[3]['median']:,.0f}")
    ax6.set_xlabel("ET Voxels per Patient", color=FG_COL)
    ax6.set_ylabel("No. of Patients", color=FG_COL)
    ax6.legend(fontsize=8, facecolor="#161b22", labelcolor=FG_COL, edgecolor="#30363d")
    ax_style(ax6, "ET (Enhancing Tumor) â€” Voxel Count Distribution")

    # â”€â”€ Plot 7: Box plots of all labels (foreground only) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ax7 = fig.add_subplot(gs[2, 0])
    bp_data   = [label_voxel_counts[l] for l in [1, 2, 3]]
    bp_labels = ["NCR", "ED", "ET"]
    bp = ax7.boxplot(
        bp_data, patch_artist=True, notch=False,
        medianprops={"color": "#f0f6fc", "lw": 2},
        whiskerprops={"color": "#8b949e"},
        capprops={"color": "#8b949e"},
        flierprops={"marker": ".", "markersize": 3, "markerfacecolor": "#8b949e", "alpha": 0.4},
    )
    for patch, col in zip(bp["boxes"], COLORS[1:]):
        patch.set_facecolor(col + "55")   # semi-transparent
        patch.set_edgecolor(col)
    ax7.set_xticklabels(bp_labels, color=FG_COL)
    ax7.set_ylabel("Voxels per Patient", color=FG_COL)
    ax_style(ax7, "Foreground Labels â€” Voxel Count Box Plots\n(whiskers = 5thâ€“95th pct)")

    # â”€â”€ Plot 8: WT / TC / ET scatter â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ax8 = fig.add_subplot(gs[2, 1])
    wt_arr = np.array(region_voxel_counts["WT"])
    tc_arr = np.array(region_voxel_counts["TC"])
    et_arr = np.array(region_voxel_counts["ET"])
    sc = ax8.scatter(wt_arr, tc_arr, c=et_arr, cmap="viridis", s=6, alpha=0.5)
    cb = fig.colorbar(sc, ax=ax8)
    cb.set_label("ET Voxels", color=FG_COL, fontsize=8)
    cb.ax.yaxis.set_tick_params(color=FG_COL, labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=FG_COL)
    ax8.set_xlabel("WT Voxels", color=FG_COL)
    ax8.set_ylabel("TC Voxels", color=FG_COL)
    ax_style(ax8, "WT vs TC (coloured by ET)\nInter-region Correlation per Patient")

    # â”€â”€ Plot 9: Imbalance ratio bar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    ax9 = fig.add_subplot(gs[2, 2])
    ratios = [
        bg_mean / max(ncr_mean, 1),
        bg_mean / max(ed_mean,  1),
        bg_mean / max(et_mean,  1),
    ]
    r_bar = ax9.barh(["BG:NCR", "BG:ED", "BG:ET"], ratios,
                     color=[COLORS[1], COLORS[2], COLORS[3]],
                     edgecolor="#30363d")
    for bar, val in zip(r_bar, ratios):
        ax9.text(val + 2, bar.get_y() + bar.get_height()/2,
                 f"{val:.0f}Ã—", va="center", color=FG_COL, fontsize=10, fontweight="bold")
    ax9.set_xlabel("Background : Foreground Ratio", color=FG_COL)
    ax_style(ax9, "Class Imbalance Ratios\n(Ã— times more BG than tumor class)")

    plot_path = OUTPUT_DIR / "dataset_analysis_plots.png"
    plt.savefig(str(plot_path), dpi=130, bbox_inches="tight", facecolor=BG_COL)
    plt.close()
    print(f"Plots saved  --> {plot_path}")

except Exception as e:
    print(f"Plot generation failed: {e}")
    import traceback; traceback.print_exc()

print("\nAll done.")

