#!/bin/bash
# =============================================================================
# SMORE Robustness vs Noise Level
# Shows MDR's advantage GROWS as noise std increases (key robustness figure).
# Training is always normal; only inference is perturbed.
#
# Methods : baseline, MDR(0.2)
# Datasets: sports, clothing
# Seeds   : 999, 42, 2024
# Modes   : noise_image, noise_both
# Std     : 0.05, 0.1, 0.2, 0.3
# Total   : 2 x 2 x 3 x 2 x 4 = 96 runs, ~22h on 2 GPUs
# =============================================================================
# NOTE: 96 runs is large. To trim, edit STDS or MODES below.
# =============================================================================

LOG_DIR="logs_noise_level_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

DATASETS=("sports" "clothing")
SEEDS=(999 42 2024)
MODES=("noise_image" "noise_both")
STDS=("0.05" "0.1" "0.2" "0.3")

METHODS=(
    "baseline|"
    "MDR|modality_dropout_rate=0.2"
)

TASKS=()
for seed in "${SEEDS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for mode in "${MODES[@]}"; do
            for std in "${STDS[@]}"; do
                for method in "${METHODS[@]}"; do
                    TASKS+=("${seed}|${dataset}|${mode}|${std}|${method}")
                done
            done
        done
    done
done
TOTAL=${#TASKS[@]}

echo "============================================================"
echo " SMORE Robustness vs Noise Level"
echo "============================================================"
echo " Log dir  : ${LOG_DIR}"
echo " Total    : ${TOTAL} runs | GPUs: 0+1 parallel"
echo " Methods  : baseline, MDR(0.2)"
echo " Datasets : ${DATASETS[*]}"
echo " Modes    : ${MODES[*]}"
echo " Stds     : ${STDS[*]}"
echo "============================================================"

PID0=0; PID1=0; COUNT=0
get_free_gpu() {
    if [ "$PID0" = "0" ] || ! kill -0 "$PID0" 2>/dev/null; then echo 0; return; fi
    if [ "$PID1" = "0" ] || ! kill -0 "$PID1" 2>/dev/null; then echo 1; return; fi
    echo -1
}
launch() {
    local g=$1 seed=$2 dataset=$3 mode=$4 std=$5 mname=$6 margs=$7
    local logfile="${LOG_DIR}/SMORE_${dataset}_${mname}_seed${seed}_${mode}_std${std}.log"
    echo "[$COUNT/$TOTAL] GPU $g | ${dataset} | ${mname} | seed=${seed} | ${mode} | std=${std} | $(date '+%H:%M:%S')"
    ( cd src && CUDA_VISIBLE_DEVICES=$g python main.py -m SMORE -d ${dataset} ${margs} seed=${seed} gpu_id=$g robust_eval_mode=${mode} robust_noise_std=${std} > "../${logfile}" 2>&1 ) &
    if [ "$g" = "0" ]; then PID0=$!; else PID1=$!; fi
}
for task in "${TASKS[@]}"; do
    COUNT=$((COUNT + 1))
    IFS='|' read -r seed dataset mode std method <<< "$task"
    IFS='|' read -r mname margs <<< "$method"
    while true; do
        g=$(get_free_gpu)
        [ "$g" != "-1" ] && { launch "$g" "$seed" "$dataset" "$mode" "$std" "$mname" "$margs"; break; }
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
echo " All ${TOTAL} noise-level runs finished at $(date '+%F %T')"
echo " python scripts/parse_smore_results.py ${LOG_DIR}/ -o results_noise_level.csv"
echo "============================================================"
