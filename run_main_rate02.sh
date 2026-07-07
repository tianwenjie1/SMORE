#!/bin/bash
# =============================================================================
# SMORE Main Table with optimal dropout rate = 0.2
# (rate search showed 0.2 > 0.1; rerun main ablation with 0.2 for consistency)
#
# Methods : baseline, FBG, MDR(0.2), FBG+MDR(0.2)
# Datasets: baby, sports, clothing
# Seeds   : 999, 42, 2024
# Total   : 4 x 3 x 3 = 36 runs, ~7h on 2 GPUs
# =============================================================================

LOG_DIR="logs_main_rate02_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

DATASETS=("baby" "sports" "clothing")
SEEDS=(999 42 2024)
RATE=0.2

METHODS=(
    "baseline|"
    "FBG|freq_band_gating=True"
    "MDR|modality_dropout_rate=${RATE}"
    "FBG+MDR|freq_band_gating=True modality_dropout_rate=${RATE}"
)

TASKS=()
for seed in "${SEEDS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for method in "${METHODS[@]}"; do
            TASKS+=("${seed}|${dataset}|${method}")
        done
    done
done
TOTAL=${#TASKS[@]}

echo "============================================================"
echo " SMORE Main Table @ dropout_rate=${RATE}"
echo "============================================================"
echo " Log dir  : ${LOG_DIR}"
echo " Total    : ${TOTAL} runs | GPUs: 0+1 parallel"
echo "============================================================"

PID0=0; PID1=0; COUNT=0
get_free_gpu() {
    if [ "$PID0" = "0" ] || ! kill -0 "$PID0" 2>/dev/null; then echo 0; return; fi
    if [ "$PID1" = "0" ] || ! kill -0 "$PID1" 2>/dev/null; then echo 1; return; fi
    echo -1
}
launch() {
    local g=$1 seed=$2 dataset=$3 mname=$4 margs=$5
    local logfile="${LOG_DIR}/SMORE_${dataset}_${mname}_seed${seed}.log"
    echo "[$COUNT/$TOTAL] GPU $g | ${dataset} | ${mname} | seed=${seed} | $(date '+%H:%M:%S')"
    ( cd src && CUDA_VISIBLE_DEVICES=$g python main.py -m SMORE -d ${dataset} ${margs} seed=${seed} gpu_id=$g > "../${logfile}" 2>&1 ) &
    if [ "$g" = "0" ]; then PID0=$!; else PID1=$!; fi
}
for task in "${TASKS[@]}"; do
    COUNT=$((COUNT + 1))
    IFS='|' read -r seed dataset method <<< "$task"
    IFS='|' read -r mname margs <<< "$method"
    while true; do
        g=$(get_free_gpu)
        [ "$g" != "-1" ] && { launch "$g" "$seed" "$dataset" "$mname" "$margs"; break; }
        sleep 15
    done
    sleep 2
done
echo ""; echo "Waiting for last runs..."
while true; do
    BUSY=0
    [ "$PID0" != "0" ] && kill -0 "$PID0" 2>/dev/null && BUSY=1
    [ "$PID1" != "0" ] && kill -0 "$PID1" 2>/dev/null && BUSY=1
    [ "$BUSY" = "0" ] && break
    sleep 15
done
echo "============================================================"
echo " All ${TOTAL} runs finished at $(date '+%F %T')"
echo " python scripts/parse_smore_results.py ${LOG_DIR}/ -o results_main_rate02.csv"
echo "============================================================"
