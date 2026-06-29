#!/bin/bash
# =============================================================================
# SMORE Innovation Points - Ablation Experiment Script
# Dual GPU Server (GPU 0 & GPU 1), nohup background execution
# =============================================================================
# Usage:
#   chmod +x run_innovations.sh
#   ./run_innovations.sh          # run all experiments
#   ./run_innovations.sh baby     # run only on baby dataset
# =============================================================================

set -e

# ----------------------------- Configuration ---------------------------------
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
echo "============================================================"
echo " SMORE Innovation Ablation Experiments"
echo "============================================================"
echo " Log directory : ${LOG_DIR}"
echo " Datasets      : ${DATASETS[*]}"
echo " Seed          : ${SEED}"
echo " GPU 0 & GPU 1 parallel execution"
echo "------------------------------------------------------------"
echo " Experiments:"
for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r name args <<< "$exp"
    printf "   %-20s %s\n" "$name" "$args"
done
echo "============================================================"
echo ""

# ----------------------------- Run Experiments --------------------------------
GPU=0
TOTAL=$((${#DATASETS[@]} * ${#EXPERIMENTS[@]}))
COUNT=0

for dataset in "${DATASETS[@]}"; do
    for exp in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r exp_name extra_args <<< "$exp"

        COUNT=$((COUNT + 1))

        # Log file naming: SMORE_{dataset}_{experiment}_{timestamp}.log
        LOG_FILE="${LOG_DIR}/SMORE_${dataset}_${exp_name}_seed${SEED}.log"

        # Build command
        CMD="cd src && nohup python main.py -m SMORE -d ${dataset} seed=${SEED} ${extra_args} > ../${LOG_FILE} 2>&1 &"

        echo "[${COUNT}/${TOTAL}] GPU ${GPU} | ${dataset} | ${exp_name}"
        echo "  Log: ${LOG_FILE}"

        # Execute with CUDA_VISIBLE_DEVICES
        CUDA_VISIBLE_DEVICES=${GPU} bash -c "${CMD}"

        # Alternate GPU
        GPU=$((1 - GPU))
    done
done

echo ""
echo "============================================================"
echo " All ${TOTAL} experiments launched in background!"
echo "============================================================"
echo ""
echo " Monitor progress:"
echo "   tail -f ${LOG_DIR}/SMORE_baby_baseline_seed${SEED}.log"
echo ""
echo " Check all running jobs:"
echo "   ps aux | grep 'main.py -m SMORE'"
echo ""
echo " Kill all SMORE experiments:"
echo "   pkill -f 'main.py -m SMORE'"
echo ""
echo " View all logs summary:"
echo "   grep 'Best results' ${LOG_DIR}/*.log"
echo ""
echo " GPU status:"
echo "   watch -n 5 nvidia-smi"
echo "============================================================"
