#!/bin/bash
# =============================================================================
# P1: MQS Baseline Scan (train-once-eval-many, paper-grade)
# Train vanilla SMORE ONCE on clean data (clean validation selects checkpoint),
# then evaluate the SAME checkpoint under all MQS modes on the test set.
#
# This is methodologically correct: "does the same clean-trained SMORE degrade
# under modality quality shift" -- NOT "different checkpoint per shift".
#
# Method  : baseline (vanilla SMORE)
# Datasets: sports, clothing
# Seeds   : 999, 42, 2024
# Train   : 2 x 3 = 6 clean trainings (save checkpoint)
# Eval    : 6 eval-only runs, each scans 11 MQS modes on the test set
# GPUs    : configurable via GPUS env var (default "2 3")
# =============================================================================

LOG_DIR="logs_mqs_baseline_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR" saved

DATASETS=("sports" "clothing")
SEEDS=(999 42 2024)

# MQS modes to evaluate on the same checkpoint
MQS_MODES="normal,shuffle_image,shuffle_text,mismatch,tail_noise_image,tail_noise_text,tail_noise_both,tail_missing_image,tail_missing_text,pop_missing_image,pop_missing_text"
# eval-time perturbation strength
EVAL_ARGS="robust_noise_std=0.3 robust_shift_ratio=0.3 robust_tail_ratio=0.3"

read -ra GPUS <<< "${GPUS:-2 3}"
NGPU=${#GPUS[@]}

# tasks: each = "train+evalscan" for one (dataset, seed)
TASKS=()
for seed in "${SEEDS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        TASKS+=("${seed}|${dataset}")
    done
done
TOTAL=${#TASKS[@]}

echo "============================================================"
echo " P1: MQS Baseline Scan (train-once-eval-many)"
echo "============================================================"
echo " Log dir  : ${LOG_DIR}"
echo " Trainings: ${TOTAL} clean baseline | each evals ${MQS_MODES}"
echo " GPUs     : ${GPUS[*]} (${NGPU}-way)"
echo "============================================================"

declare -a PID
for ((i=0; i<NGPU; i++)); do PID[$i]=0; done
get_free_slot() {
    for ((i=0; i<NGPU; i++)); do
        if [ "${PID[$i]}" = "0" ] || ! kill -0 "${PID[$i]}" 2>/dev/null; then echo $i; return; fi
    done
    echo -1
}

COUNT=0
for task in "${TASKS[@]}"; do
    COUNT=$((COUNT + 1))
    IFS='|' read -r seed dataset <<< "$task"
    while true; do
        slot=$(get_free_slot)
        if [ "$slot" != "-1" ]; then
            g=${GPUS[$slot]}
            logfile="${LOG_DIR}/SMORE_${dataset}_baseline_seed${seed}.log"
            ckpt="saved/SMORE-${dataset}-seed${seed}-baseline.pt"
            echo "[$COUNT/$TOTAL] GPU $g | ${dataset} | baseline | seed=${seed} | train+evalscan | $(date '+%H:%M:%S')"
            # train clean (saves checkpoint under src/saved/), then eval-only scan
            # on the same ckpt. Both run from src/, so ckpt path is relative to src/.
            ( cd src && \
              CUDA_VISIBLE_DEVICES=$g python main.py -m SMORE -d ${dataset} seed=${seed} gpu_id=$g ckpt_tag=baseline robust_eval_mode=normal > "../${logfile}" 2>&1 && \
              CUDA_VISIBLE_DEVICES=$g python main.py -m SMORE -d ${dataset} seed=${seed} gpu_id=$g ckpt_tag=baseline ${EVAL_ARGS} --eval-only --ckpt "${ckpt}" --eval-modes ${MQS_MODES} >> "../${logfile}" 2>&1 ) &
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
echo " All ${TOTAL} train+evalscan finished at $(date '+%F %T')"
echo " python scripts/parse_smore_results.py ${LOG_DIR}/ -o results_mqs_baseline.csv"
echo "============================================================"
