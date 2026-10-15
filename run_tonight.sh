#!/bin/bash
# =============================================================================
# run_tonight.sh — Single-Command Overnight Pipeline
# Brain Tumor Segmentation | RCOEM 2026–27 | DGX B200
#
# WHAT THIS DOES (automatically in order):
#   1. Pull latest code from GitHub
#   2. Generate all visualisation figures (attention maps, overlays, error maps)
#   3. Train the 2 missing ablation variants (channel-only SE & spatial-only gates)
#   4. Evaluate all 4 ablation models on the held-out test set
#   5. Generate ablation table CSV + bar chart
#
# HOW TO RUN (one command — then you can close your laptop!):
#   chmod +x run_tonight.sh
#   nohup bash run_tonight.sh > run_tonight.log 2>&1 &
#
# HOW TO WATCH PROGRESS:
#   tail -f run_tonight.log
#
# TOMORROW MORNING — everything will be in:
#   results/figures/        ← all attention heatmaps + overlays
#   results/ablation_study.csv   ← 4-model comparison table
#   results/ablation_barchart.png ← bar chart for your PPT
# =============================================================================

set -e  # Stop if any command fails

LOG_PREFIX="[run_tonight.sh]"
START_TIME=$(date)

echo ""
echo "=============================================================="
echo "  BraTS 2023 Overnight Pipeline"
echo "  Started at: $START_TIME"
echo "=============================================================="

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_DIR=~/Brain_Tumor_Segmentation
cd "$PROJECT_DIR"

# ── Environment ────────────────────────────────────────────────────────────
export PYTORCH_NO_CUDA_NVML=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Activate venv if it exists
if [ -d "$PROJECT_DIR/venv" ]; then
    source "$PROJECT_DIR/venv/bin/activate"
    echo "$LOG_PREFIX Activated venv"
fi

# =============================================================================
# STEP 1 — Pull latest code
# =============================================================================
echo ""
echo "=============================================================="
echo "  STEP 1 / 3 — Pulling latest code from GitHub"
echo "=============================================================="
git pull origin server-progress
echo "$LOG_PREFIX Code up to date."

# =============================================================================
# STEP 2 — Visualise the 2 trained models (baseline + attention)
# Generates: results/figures/overlay_*.png
#            results/figures/attention_heatmaps_*.png
#            results/figures/error_analysis_*.png
#            results/figures/master_qualitative_comparison.png
# Estimated time: ~10–20 minutes
# =============================================================================
echo ""
echo "=============================================================="
echo "  STEP 2 / 3 — Generating Visualisations (~10–20 minutes)"
echo "  Outputs: results/figures/"
echo "=============================================================="
STEP2_START=$(date +%s)

python3 visualise_results.py

STEP2_END=$(date +%s)
STEP2_MINS=$(( (STEP2_END - STEP2_START) / 60 ))
echo "$LOG_PREFIX ✅ Visualisation complete in ${STEP2_MINS} minutes."
echo "$LOG_PREFIX Figures saved in: $PROJECT_DIR/results/figures/"

# =============================================================================
# STEP 3 — Ablation Study
# Trains channel_only and spatial_only if not already trained,
# then evaluates all 4 models and generates:
#   results/ablation_study.csv
#   results/ablation_barchart.png
# Estimated time: ~5–7 hours
# =============================================================================
echo ""
echo "=============================================================="
echo "  STEP 3 / 3 — Running Ablation Study (~5–7 hours)"
echo "  This trains Channel-Only (SE) and Spatial-Only (Gates)"
echo "  then evaluates all 4 models on 122 held-out test patients"
echo "=============================================================="
STEP3_START=$(date +%s)

python3 run_ablation.py --train-missing

STEP3_END=$(date +%s)
STEP3_MINS=$(( (STEP3_END - STEP3_START) / 60 ))
echo "$LOG_PREFIX ✅ Ablation study complete in ${STEP3_MINS} minutes."
echo "$LOG_PREFIX Results saved in: $PROJECT_DIR/results/"

# =============================================================================
# DONE!
# =============================================================================
END_TIME=$(date)
TOTAL_SECS=$(( STEP2_END - STEP2_START + STEP3_END - STEP3_START ))
TOTAL_MINS=$(( TOTAL_SECS / 60 ))

echo ""
echo "=============================================================="
echo "  ✅ ALL DONE! Everything is ready for your presentation."
echo ""
echo "  Started at : $START_TIME"
echo "  Finished at: $END_TIME"
echo "  Total time : ~${TOTAL_MINS} minutes"
echo ""
echo "  📁 Files ready for your PPT:"
echo "     results/figures/overlay_*.png"
echo "     results/figures/attention_heatmaps_*.png"
echo "     results/figures/error_analysis_*.png"
echo "     results/figures/master_qualitative_comparison.png"
echo "     results/ablation_study.csv"
echo "     results/ablation_barchart.png"
echo "=============================================================="
