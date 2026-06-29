#!/bin/bash
# =============================================================================
# SMORE Innovation Points - Ablation Experiment Script
# Dual GPU Server (GPU 0 & GPU 1), 2 experiments in parallel at a time
# Each GPU runs only ONE experiment at a time to avoid CUBLAS errors
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

# ----------------------------- Build task queue ------------------------------
TASKS=()
for dataset in "${DATASETS[@]}"; do
    for exp in "${EXPERIMENTS[@]}"; do
        IFS='|' read -r exp_name extra_args <<< "$exp"
        TASKS+=("${dataset}|${exp_name}|${extra_args}")
    done
done

TOTAL=${#TASKS[@]}

# ----------------------------- Print Summary ---------------------------------
echo "============================================================"
echo " SMORE Innovation Ablation Experiments"
echo "============================================================"
echo " Log directory : ${LOG_DIR}"
echo " Total tasks   : ${TOTAL}"
echo " Parallelism   : 2 (GPU 0 + GPU 1)"
echo "------------------------------------------------------------"
echo " Experiments:"
for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r name args <<< "$exp"
    printf "   %-20s %s\n" "$name" "$args"
done
echo "============================================================"
echo ""

# ----------------------------- Run with 2 GPU slots --------------------------
GPU0_PID=""
GPU1_PID=""
GPU0_LOG=""
GPU1_LOG=""
COUNT=0

wait_for_slot() {
    # Wait until at least one GPU slot is free
    while true; do
        SLOT_FREE=-1
        if [ -z "$GPU0_PID" ] || ! kill -0 "$GPU0_PID" 2>/dev/null; then
            SLOT_FREE=0
            GPU0_PID=""
        fi
        if [ -z "$GPU1_PID" ] || ! kill -0 "$GPU1_PID" 2>/dev/null; then
            SLOT_FREE=1
            GPU1_PID=""
        fi
        if [ "$SLOT_FREE" -ne -1 ]; then
            return $SLOT_FREE
        fi
        sleep 10
    done
}

launch_task() {
    local GPU_ID=$1
    local DATASET=$2
    local EXP_NAME=$3
    local EXTRA_ARGS=$4
    local LOG_FILE="${LOG_DIR}/SMORE_${DATASET}_${EXP_NAME}_seed${SEED}.log"

    echo "[${COUNT}/${TOTAL}] GPU ${GPU_ID} | ${DATASET} | ${EXP_NAME} | started at $(date +%H:%M:%S)"
    echo "  Log: ${LOG_FILE}"

    CUDA_VISIBLE_DEVICES=${GPU_ID} nohup python src/main.py -m SMORE -d ${DATASET} seed=${SEED} ${EXTRA_ARGS} > "${LOG_FILE}" 2>&1 &
    local PID=$!

    if [ "$GPU_ID" -eq 0 ]; then
        GPU0_PID=$PID
        GPU0_LOG=$LOG_FILE
    else
        GPU1_PID=$PID
        GPU1_LOG=$LOG_FILE
    fi
}

# Launch tasks, 2 at a time
for task in "${TASKS[@]}"; do
    COUNT=$((COUNT + 1))
    IFS='|' read -r dataset exp_name extra_args <<< "$task"

    wait_for_slot
    GPU_ID=$?
    launch_task $GPU_ID "$dataset" "$exp_name" "$extra_args"
done

# Wait for remaining tasks to finish
echo ""
echo "All tasks launched. Waiting for remaining to finish..."
while true; do
    RUNNING=0
    [ -n "$GPU0_PID" ] && kill -0 "$GPU0_PID" 2>/dev/null && RUNNING=$((RUNNING + 1))
    [ -n "$GPU1_PID" ] && kill -0 "$GPU1_PID" 2>/dev/null && RUNNING=$((RUNNING + 1))
    if [ "$RUNNING" -eq 0 ]; then
        break
    fi
    sleep 10
done

echo ""
echo "============================================================"
echo " All ${TOTAL} experiments completed!"
echo "============================================================"
echo ""
echo " View results:"
echo "   grep 'Best results' ${LOG_DIR}/*.log"
echo ""
echo " Check for errors:"
echo "   grep -l 'Error' ${LOG_DIR}/*.log"
echo "============================================================"
