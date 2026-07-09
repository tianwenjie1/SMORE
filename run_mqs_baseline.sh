#!/bin/bash
# =============================================================================
# P1: MQS Baseline Scan
# Run vanilla SMORE under all Modality Quality Shift scenarios to PROVE the
# problem exists (SMORE is unstable under modality quality shift).
#
# Method  : baseline (vanilla SMORE, no innovations)
# Datasets: sports, clothing
# Seeds   : 999, 42, 2024
# Modes   : normal + 10 MQS shift types (fixed ratio=0.3, tail-noise std=0.3)
# Total   : 11 x 2 x 3 = 66 runs
# GPUs    : configurable via GPUS env var (default: "2 3")
# =============================================================================
# NOTE: noise-std sweep already covered by run_noise_level.sh; here we focus on
# the NEW shift types (shuffle / mismatch / tail / pop-missing).
# =============================================================================

LOG_DIR="logs_mqs_baseline_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

DATASETS=("sports" "clothing")
SEEDS=(999 42 2024)

# mode | extra_args
MQS_MODES=(
    "normal|"
    "shuffle_image|robust_shift_ratio=0.3"
    "shuffle_text|robust_shift_ratio=0.3"
    "mismatch|robust_shift_ratio=0.3"
    "tail_noise_image|robust_noise_std=0.3"
    "tail_noise_text|robust_noise_std=0.3"
    "tail_noise_both|robust_noise_std=0.3"
    "tail_missing_image|"
    "tail_missing_text|"
    "pop_missing_image|robust_shift_ratio=0.3"
    "pop_missing_text|robust_shift_ratio=0.3"
)

read -ra GPUS <<< "${GPUS:-2 3}"
NGPU=${#GPUS[@]}

TASKS=()
for seed in "${SEEDS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for mq in "${MQS_MODES[@]}"; do
            TASKS+=("${seed}|${dataset}|${mq}")
        done
    done
done
TOTAL=${#TASKS[@]}

echo "============================================================"
echo " P1: MQS Baseline Scan (prove SMORE unstable under MQS)"
echo "============================================================"
echo " Log dir  : ${LOG_DIR}"
echo " Total    : ${TOTAL} runs | GPUs: ${GPUS[*]} (${NGPU}-way)"
echo " Datasets : ${DATASETS[*]}"
echo "============================================================"

declare -a PID
for ((i=0; i<NGPU; i++)); do PID[$i]=0; done

get_free_slot() {
    for ((i=0; i<NGPU; i++)); do
        if [ "${PID[$i]}" = "0" ] || ! kill -0 "${PID[$i]}" 2>/dev/null; then
            echo $i; return
        fi
    done
    echo -1
}

COUNT=0
for task in "${TASKS[@]}"; do
    COUNT=$((COUNT + 1))
    IFS='|' read -r seed dataset mq <<< "$task"
    IFS='|' read -r mode margs <<< "$mq"
    while true; do
        slot=$(get_free_slot)
        if [ "$slot" != "-1" ]; then
            g=${GPUS[$slot]}
            logfile="${LOG_DIR}/SMORE_${dataset}_baseline_seed${seed}_${mode}.log"
            echo "[$COUNT/$TOTAL] GPU $g | ${dataset} | baseline | seed=${seed} | ${mode} | $(date '+%H:%M:%S')"
            ( cd src && CUDA_VISIBLE_DEVICES=$g python main.py -m SMORE -d ${dataset} seed=${seed} gpu_id=$g robust_eval_mode=${mode} ${margs} > "../${logfile}" 2>&1 ) &
            PID[$slot]=$!
            break
        fi
        sleep 15
    done
    sleep 2
done

echo ""; echo "Waiting for last runs..."
while true; do
    BUSY=0
    for ((i=0; i<NGPU; i++)); do
        [ "${PID[$i]}" != "0" ] && kill -0 "${PID[$i]}" 2>/dev/null && BUSY=1
    done
    [ "$BUSY" = "0" ] && break
    sleep 15
done
echo "============================================================"
echo " All ${TOTAL} MQS baseline runs finished at $(date '+%F %T')"
echo " python scripts/parse_smore_results.py ${LOG_DIR}/ -o results_mqs_baseline.csv"
echo "============================================================"
