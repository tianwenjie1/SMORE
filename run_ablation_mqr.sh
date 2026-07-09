#!/bin/bash
# =============================================================================
# P3: MQR ablation (train-once-eval-many, paper-grade)
# Each method trained ONCE on clean data (clean validation selects checkpoint),
# then the SAME checkpoint evaluated under key MQS modes.
#
# Methods (7): baseline / dropout / noise_aug / mqr_bpr / mqr_ps / mqr_full / mqr_full_tail
# Datasets : sports, clothing
# Seeds    : 999, 42, 2024
# Train    : 7 x 2 x 3 = 42 clean trainings (save checkpoint per method)
# Eval     : 42 eval-only scans, each over 4 MQS modes on the test set
# GPUs     : configurable via GPUS env var (default "2 3")
# =============================================================================
# 命门: mqr_full_tail must clearly beat dropout & noise_aug under MQS modes,
#       otherwise MQR is still a dropout trick.
# =============================================================================

LOG_DIR="logs_ablation_mqr_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR" saved

DATASETS=("sports" "clothing")
SEEDS=(999 42 2024)

# MQS modes to evaluate on each checkpoint
MQS_MODES="normal,tail_noise_both,mismatch,pop_missing_image"
EVAL_ARGS="robust_noise_std=0.3 robust_shift_ratio=0.3 robust_tail_ratio=0.3"

# method_name | train args
METHODS=(
    "baseline|"
    "dropout|modality_dropout_rate=0.2"
    "noise_aug|train_noise_std=0.3"
    "mqr_bpr|mqr_enabled=True mqr_alpha=0.5 mqr_beta=0.0"
    "mqr_ps|mqr_enabled=True mqr_alpha=0.0 mqr_beta=0.2"
    "mqr_full|mqr_enabled=True mqr_alpha=0.5 mqr_beta=0.2 mqr_tail_weight=False"
    "mqr_full_tail|mqr_enabled=True mqr_alpha=0.5 mqr_beta=0.2 mqr_tail_weight=True"
)

read -ra GPUS <<< "${GPUS:-2 3}"
NGPU=${#GPUS[@]}

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
echo " P3: MQR ablation (train-once-eval-many)"
echo "============================================================"
echo " Log dir  : ${LOG_DIR}"
echo " Trainings: ${TOTAL} | each evals ${MQS_MODES}"
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
    IFS='|' read -r seed dataset method <<< "$task"
    IFS='|' read -r mname margs <<< "$method"
    while true; do
        slot=$(get_free_slot)
        if [ "$slot" != "-1" ]; then
            g=${GPUS[$slot]}
            logfile="${LOG_DIR}/SMORE_${dataset}_${mname}_seed${seed}.log"
            ckpt="saved/SMORE-${dataset}-seed${seed}-${mname}.pt"
            echo "[$COUNT/$TOTAL] GPU $g | ${dataset} | ${mname} | seed=${seed} | train+evalscan | $(date '+%H:%M:%S')"
            ( cd src && \
              CUDA_VISIBLE_DEVICES=$g python main.py -m SMORE -d ${dataset} ${margs} seed=${seed} gpu_id=$g ckpt_tag=${mname} robust_eval_mode=normal > "../${logfile}" 2>&1 && \
              CUDA_VISIBLE_DEVICES=$g python main.py -m SMORE -d ${dataset} seed=${seed} gpu_id=$g ${EVAL_ARGS} --eval-only --ckpt "../${ckpt}" --eval-modes ${MQS_MODES} >> "../${logfile}" 2>&1 ) &
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
echo " python scripts/parse_smore_results.py ${LOG_DIR}/ -o results_ablation_mqr.csv"
echo "============================================================"
