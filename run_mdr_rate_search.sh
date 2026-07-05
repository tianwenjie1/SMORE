#!/bin/bash
# =============================================================================
# SMORE MDR Dropout Rate Search
# Search modality_dropout_rate on sports + clothing (the two datasets where
# MDR/FBG+MDR showed clear gains).
#
# Methods  : MDR, FBG+MDR
# Datasets : sports, clothing
# Rates    : 0.05, 0.1, 0.2, 0.3
# Seeds    : 999, 42, 2024
# Total    : 2 x 2 x 4 x 3 = 48 runs
# GPUs     : 0 + 1 (one process each, parallel)
# =============================================================================

LOG_DIR="logs_mdr_rate_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

DATASETS=("sports" "clothing")
SEEDS=(999 42 2024)
RATES=("0.05" "0.1" "0.2" "0.3")

# method_name | extra_args (modality_dropout_rate appended per-run)
METHODS=(
    "MDR|freq_band_gating=False"
    "FBG+MDR|freq_band_gating=True"
)

# ----------------------- Build task queue ------------------------------------
TASKS=()
for seed in "${SEEDS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for rate in "${RATES[@]}"; do
            for method in "${METHODS[@]}"; do
                TASKS+=("${seed}|${dataset}|${rate}|${method}")
            done
        done
    done
done
TOTAL=${#TASKS[@]}

echo "============================================================"
echo " SMORE MDR Dropout Rate Search"
echo "============================================================"
echo " Log dir  : ${LOG_DIR}"
echo " Total    : ${TOTAL} runs"
echo " GPUs     : 0 + 1 (one process each, parallel)"
echo " Methods  : MDR, FBG+MDR"
echo " Datasets : ${DATASETS[*]}"
echo " Rates    : ${RATES[*]}"
echo " Seeds    : ${SEEDS[*]}"
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
    local g=$1 seed=$2 dataset=$3 rate=$4 mname=$5 margs=$6
    local logfile="${LOG_DIR}/SMORE_${dataset}_${mname}_rate${rate}_seed${seed}.log"
    echo "[$COUNT/$TOTAL] GPU $g | ${dataset} | ${mname} | rate=${rate} | seed=${seed} | $(date '+%H:%M:%S')"
    ( cd src && CUDA_VISIBLE_DEVICES=$g python main.py -m SMORE -d ${dataset} ${margs} modality_dropout_rate=${rate} seed=${seed} gpu_id=$g > "../${logfile}" 2>&1 ) &
    if [ "$g" = "0" ]; then PID0=$!; else PID1=$!; fi
}

for task in "${TASKS[@]}"; do
    COUNT=$((COUNT + 1))
    IFS='|' read -r seed dataset rate method <<< "$task"
    IFS='|' read -r mname margs <<< "$method"
    while true; do
        g=$(get_free_gpu)
        if [ "$g" != "-1" ]; then
            launch "$g" "$seed" "$dataset" "$rate" "$mname" "$margs"
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
echo " All ${TOTAL} rate-search runs finished at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo " Parse results to CSV:"
echo "   python scripts/parse_smore_results.py ${LOG_DIR} -o results_mdr_rate.csv"
echo "============================================================"
