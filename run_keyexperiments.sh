#!/bin/bash
# =============================================================================
# SMORE Key Innovations - Comprehensive Multi-Seed Evaluation
# Dual GPU parallel (ONE process per GPU -> safe, no CUBLAS/OOM)
# Methods : baseline, FBG, MDR, FBG+MDR
# Datasets: baby, sports, clothing
# Seeds   : 999, 42, 2024  (for mean +/- std reporting)
# Total   : 4 x 3 x 3 = 36 runs, ~7-8 hours on 2 GPUs
# =============================================================================

LOG_DIR="logs_keyexperiments_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

DATASETS=("baby" "sports" "clothing")
SEEDS=(999 42 2024)

# method_name | extra_args
METHODS=(
    "baseline|"
    "FBG|freq_band_gating=True"
    "MDR|modality_dropout_rate=0.1"
    "FBG+MDR|freq_band_gating=True modality_dropout_rate=0.1"
)

# ----------------------- Build task queue ------------------------------------
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
echo " SMORE Key Innovations - Multi-Seed Evaluation"
echo "============================================================"
echo " Log dir  : ${LOG_DIR}"
echo " Total    : ${TOTAL} runs"
echo " GPUs     : 0 + 1 (one process each, parallel)"
echo " Methods  : baseline, FBG, MDR, FBG+MDR"
echo " Datasets : baby, sports, clothing"
echo " Seeds    : 999, 42, 2024"
echo "============================================================"
echo ""

# ----------------------- 2-GPU scheduler -------------------------------------
PID0=0
PID1=0
COUNT=0

get_free_gpu() {
    # returns 0, 1, or -1 (none free)
    if [ "$PID0" = "0" ] || ! kill -0 "$PID0" 2>/dev/null; then
        echo 0; return
    fi
    if [ "$PID1" = "0" ] || ! kill -0 "$PID1" 2>/dev/null; then
        echo 1; return
    fi
    echo -1
}

launch() {
    local g=$1
    local seed=$2
    local dataset=$3
    local mname=$4
    local margs=$5
    local logfile="${LOG_DIR}/SMORE_${dataset}_${mname}_seed${seed}.log"
    echo "[$COUNT/$TOTAL] GPU $g | ${dataset} | ${mname} | seed=${seed} | started $(date '+%H:%M:%S')"
    # subshell isolates cwd so cd src doesn't race between launches
    ( cd src && CUDA_VISIBLE_DEVICES=$g python main.py -m SMORE -d ${dataset} ${margs} seed=${seed} gpu_id=$g > "../${logfile}" 2>&1 ) &
    if [ "$g" = "0" ]; then PID0=$!; else PID1=$!; fi
}

for task in "${TASKS[@]}"; do
    COUNT=$((COUNT + 1))
    IFS='|' read -r seed dataset method <<< "$task"
    IFS='|' read -r mname margs <<< "$method"

    # wait until a GPU slot is free
    while true; do
        g=$(get_free_gpu)
        if [ "$g" != "-1" ]; then
            launch "$g" "$seed" "$dataset" "$mname" "$margs"
            break
        fi
        sleep 15
    done
    # tiny pause so the launched process registers before we check again
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
echo " All ${TOTAL} runs finished at $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""
echo " Quick result summary:"
echo "   for f in ${LOG_DIR}/SMORE_*.log; do"
echo "     name=\$(basename \"\$f\" .log | sed 's/SMORE_//')"
echo "     grep \"Test:\" \"\$f\" | tail -1 | sed \"s/^/\$name | /\""
echo "   done"
echo "============================================================"
