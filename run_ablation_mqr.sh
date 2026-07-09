#!/bin/bash
# =============================================================================
# P3: 命门消融 — prove MQR is NOT plain dropout / noise augmentation
# Trains each variant once, evaluates under clean + key MQS scenarios.
#
# Methods (7):
#   1. baseline                       (vanilla SMORE)
#   2. +dropout                       (modality_dropout_rate=0.2, naive)
#   3. +noise_aug                     (train_noise_std=0.3, naive)
#   4. +mqr_bpr                       (mqr_enabled, beta=0 : degraded-BPR only)
#   5. +mqr_ps                        (mqr_enabled, alpha=0 : PS-loss only)
#   6. +mqr_full                      (mqr_enabled, alpha=0.5, beta=0.2, no tail)
#   7. +mqr_full_tail                 (complete method, tail-weighted)
# Datasets : sports, clothing
# Seeds    : 999, 42, 2024
# Eval     : normal + tail_noise_both(std0.3) + mismatch(r0.3)
# Total    : 7 x 2 x 3 x 3 = 126 runs
# GPUs     : configurable via GPUS env var (default "2 3")
# =============================================================================
# Trim: set SEEDS=(999) and EVAL_MODES=("normal" "tail_noise_both") -> 28 runs
# =============================================================================

LOG_DIR="logs_ablation_mqr_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

DATASETS=("sports" "clothing")
SEEDS=(999 42 2024)
# eval mode | extra eval args
EVAL_MODES=(
    "normal|"
    "tail_noise_both|robust_noise_std=0.3"
    "mismatch|robust_shift_ratio=0.3"
)

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
            for ev in "${EVAL_MODES[@]}"; do
                TASKS+=("${seed}|${dataset}|${method}|${ev}")
            done
        done
    done
done
TOTAL=${#TASKS[@]}

echo "============================================================"
echo " P3: MQR ablation (prove not-dropout)"
echo "============================================================"
echo " Log dir  : ${LOG_DIR}"
echo " Total    : ${TOTAL} runs | GPUs: ${GPUS[*]} (${NGPU}-way)"
echo " Methods  : 7 | Eval modes: normal + tail_noise_both + mismatch"
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
    IFS='|' read -r seed dataset method ev <<< "$task"
    IFS='|' read -r mname margs <<< "$method"
    IFS='|' read -r emode eargs <<< "$ev"
    while true; do
        slot=$(get_free_slot)
        if [ "$slot" != "-1" ]; then
            g=${GPUS[$slot]}
            logfile="${LOG_DIR}/SMORE_${dataset}_${mname}_seed${seed}_${emode}.log"
            echo "[$COUNT/$TOTAL] GPU $g | ${dataset} | ${mname} | seed=${seed} | eval=${emode} | $(date '+%H:%M:%S')"
            ( cd src && CUDA_VISIBLE_DEVICES=$g python main.py -m SMORE -d ${dataset} ${margs} seed=${seed} gpu_id=$g robust_eval_mode=${emode} ${eargs} > "../${logfile}" 2>&1 ) &
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
echo " All ${TOTAL} ablation runs finished at $(date '+%F %T')"
echo " python scripts/parse_smore_results.py ${LOG_DIR}/ -o results_ablation_mqr.csv"
echo "============================================================"
