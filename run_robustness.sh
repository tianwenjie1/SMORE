#!/bin/bash
# =============================================================================
# SMORE Robustness Evaluation
# Tests model performance under inference-time modality perturbation.
# Training is ALWAYS normal; only evaluation is perturbed.
#
# Methods   : baseline, MDR, FBG+MDR
# Datasets  : baby, sports, clothing
# Seeds     : 999, 42, 2024
# Robust    : normal, drop_image, drop_text, noise_image, noise_text, noise_both
# Total     : 3 x 3 x 3 x 6 = 162 runs
# GPUs      : 0 + 1 (one process each, parallel)
# =============================================================================
# NOTE: 162 runs is large. To run a subset, edit DATASETS / SEEDS / MODES below.
# =============================================================================

LOG_DIR="logs_robustness_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

DATASETS=("baby" "sports" "clothing")
SEEDS=(999 42 2024)
MODES=("normal" "drop_image" "drop_text" "noise_image" "noise_text" "noise_both")

# method_name | extra_args
METHODS=(
    "baseline|"
    "MDR|modality_dropout_rate=0.1"
    "FBG+MDR|freq_band_gating=True modality_dropout_rate=0.1"
)

# ----------------------- Build task queue ------------------------------------
TASKS=()
for seed in "${SEEDS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for mode in "${MODES[@]}"; do
            for method in "${METHODS[@]}"; do
                TASKS+=("${seed}|${dataset}|${mode}|${method}")
            done
        done
    done
done
TOTAL=${#TASKS[@]}

echo "============================================================"
echo " SMORE Robustness Evaluation"
echo "============================================================"
echo " Log dir  : ${LOG_DIR}"
echo " Total    : ${TOTAL} runs"
echo " GPUs     : 0 + 1 (one process each, parallel)"
echo " Methods  : baseline, MDR, FBG+MDR"
echo " Datasets : ${DATASETS[*]}"
echo " Seeds    : ${SEEDS[*]}"
echo " Modes    : ${MODES[*]}"
echo "============================================================"
echo ""

# ----------------------- 2-GPU scheduler -------------------------------------
PID0=0
PID1=0
COUNT=0

get_free_gpu() {
    if [ "$PID0" = "0" ] || ! kill -0 "$PID0" 2>/dev/null; then echo 0; return; fi
    if [ "$PID1" = "0" ] || ! kill -0 "$PID1" 2>/dev/null; then echo 1; return; fi
    echo -1
}

launch() {
    local g=$1 seed=$2 dataset=$3 mode=$4 mname=$5 margs=$6
    local logfile="${LOG_DIR}/SMORE_${dataset}_${mname}_seed${seed}_${mode}.log"
    echo "[$COUNT/$TOTAL] GPU $g | ${dataset} | ${mname} | seed=${seed} | ${mode} | $(date '+%H:%M:%S')"
    ( cd src && CUDA_VISIBLE_DEVICES=$g python main.py -m SMORE -d ${dataset} ${margs} seed=${seed} robust_eval_mode=${mode} > "../${logfile}" 2>&1 ) &
    if [ "$g" = "0" ]; then PID0=$!; else PID1=$!; fi
}

for task in "${TASKS[@]}"; do
    COUNT=$((COUNT + 1))
    IFS='|' read -r seed dataset mode method <<< "$task"
    IFS='|' read -r mname margs <<< "$method"
    while true; do
        g=$(get_free_gpu)
        if [ "$g" != "-1" ]; then
            launch "$g" "$seed" "$dataset" "$mode" "$mname" "$margs"
            break
        fi
        sleep 15
    done
    sleep 2
done

# ----------------------- Wait for remaining ----------------------------------
echo ""
echo "All tasks launched. Waiting for the last ones to finish..."
while true; do
    BUSY=0
    [ "$PID0" != "0" ] && kill -0 "$PID0" 2>/dev/null && BUSY=1
    [ "$PID1" != "0" ] && kill -0 "$PID1" 2>/dev/null && BUSY=1
    [ "$BUSY" = "0" ] && break
    sleep 15
done

echo ""
echo "============================================================"
echo " All ${TOTAL} robustness runs finished at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo " Parse results to CSV:"
echo "   python scripts/parse_smore_results.py ${LOG_DIR} -o results_robustness.csv"
echo "============================================================"
