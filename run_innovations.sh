#!/bin/bash
# =============================================================================
# SMORE Innovation Points - Ablation Experiment Script
# Single GPU, one experiment at a time, sequential execution
# =============================================================================

set -e

# ----------------------------- Configuration ---------------------------------
GPU_ID=0                          # Which GPU to use (0 or 1)
LOG_DIR="logs_innovations_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

DATASETS=("baby" "sports" "clothing")
SEED=999

# Innovation experiment definitions
# Format: "experiment_name|extra_args"
EXPERIMENTS=(
    "baseline|"
    "FBG|freq_band_gating=True"
    "MRG|modality_reliability_gating=True"
    "MDR|modality_dropout_rate=0.1"
    "GER|graph_edge_reweighting=True"
    "FBG+MRG|freq_band_gating=True modality_reliability_gating=True"
    "FBG+MRG+MDR|freq_band_gating=True modality_reliability_gating=True modality_dropout_rate=0.1"
    "FBG+MRG+MDR+GER|freq_band_gating=True modality_reliability_gating=True modality_dropout_rate=0.1 graph_edge_reweighting=True"
)

# ----------------------------- Filter datasets -------------------------------
if [ $# -gt 0 ]; then
    DATASETS=("$@")
fi

# ----------------------------- Print Summary ---------------------------------
TOTAL=$((${#DATASETS[@]} * ${#EXPERIMENTS[@]}))

echo "============================================================"
echo " SMORE Innovation Ablation Experiments"
echo "============================================================"
echo " GPU           : ${GPU_ID}"
echo " Log directory : ${LOG_DIR}"
echo " Total tasks   : ${TOTAL}"
echo " Mode          : Sequential (one at a time)"
echo "------------------------------------------------------------"
echo " Experiments:"
for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r name args <<< "$exp"
    printf "   %-20s %s\n" "$name" "$args"
done
echo "============================================================"
echo ""

# ----------------------------- Run experiments sequentially -------------------
COUNT=0

for dataset in "${DATASETS[@]}"; do
    for exp in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r exp_name extra_args <<< "$exp"
        COUNT=$((COUNT + 1))

        LOG_FILE="${LOG_DIR}/SMORE_${dataset}_${exp_name}_seed${SEED}.log"

        echo "[${COUNT}/${TOTAL}] GPU ${GPU_ID} | ${dataset} | ${exp_name} | started at $(date '+%Y-%m-%d %H:%M:%S')"

        CUDA_VISIBLE_DEVICES=${GPU_ID} python src/main.py -m SMORE -d ${dataset} seed=${SEED} ${extra_args} > "${LOG_FILE}" 2>&1
        EXIT_CODE=$?

        if [ ${EXIT_CODE} -eq 0 ]; then
            echo "[${COUNT}/${TOTAL}] ${dataset} | ${exp_name} | DONE ✓"
        else
            echo "[${COUNT}/${TOTAL}] ${dataset} | ${exp_name} | FAILED ✗ (exit code ${EXIT_CODE})"
        fi
        echo ""
    done
done

echo "============================================================"
echo " All ${TOTAL} experiments finished!"
echo "============================================================"
echo ""
echo " Results summary:"
echo "   grep 'Best results' ${LOG_DIR}/*.log"
echo ""
echo " Errors check:"
echo "   grep -l 'Error' ${LOG_DIR}/*.log"
echo "============================================================"
