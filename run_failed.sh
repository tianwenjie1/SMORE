#!/bin/bash
# =============================================================================
# Re-run only the FAILED experiments (MRG-related + GER on clothing/sports)
# =============================================================================
GPU_ID=0
LOG_DIR="logs_rerun_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
SEED=999

# Only the experiments that failed due to bugs
TASKS=(
    "baby|MRG|modality_reliability_gating=True"
    "baby|FBG+MRG|freq_band_gating=True modality_reliability_gating=True"
    "baby|FBG+MRG+MDR|freq_band_gating=True modality_reliability_gating=True modality_dropout_rate=0.1"
    "baby|FBG+MRG+MDR+GER|freq_band_gating=True modality_reliability_gating=True modality_dropout_rate=0.1 graph_edge_reweighting=True"
    "sports|MRG|modality_reliability_gating=True"
    "sports|GER|graph_edge_reweighting=True"
    "sports|FBG+MRG|freq_band_gating=True modality_reliability_gating=True"
    "sports|FBG+MRG+MDR|freq_band_gating=True modality_reliability_gating=True modality_dropout_rate=0.1"
    "sports|FBG+MRG+MDR+GER|freq_band_gating=True modality_reliability_gating=True modality_dropout_rate=0.1 graph_edge_reweighting=True"
    "clothing|MRG|modality_reliability_gating=True"
    "clothing|GER|graph_edge_reweighting=True"
    "clothing|FBG+MRG|freq_band_gating=True modality_reliability_gating=True"
    "clothing|FBG+MRG+MDR|freq_band_gating=True modality_reliability_gating=True modality_dropout_rate=0.1"
    "clothing|FBG+MRG+MDR+GER|freq_band_gating=True modality_reliability_gating=True modality_dropout_rate=0.1 graph_edge_reweighting=True"
)

TOTAL=${#TASKS[@]}
COUNT=0

echo "============================================================"
echo " Re-run FAILED experiments (${TOTAL} tasks)"
echo " GPU: ${GPU_ID} | Log: ${LOG_DIR}"
echo "============================================================"

for task in "${TASKS[@]}"; do
    IFS='|' read -r dataset exp_name extra_args <<< "$task"
    COUNT=$((COUNT + 1))
    LOG_FILE="${LOG_DIR}/SMORE_${dataset}_${exp_name}_seed${SEED}.log"
    echo "[${COUNT}/${TOTAL}] GPU ${GPU_ID} | ${dataset} | ${exp_name} | started at $(date '+%H:%M:%S')"
    cd src
    CUDA_VISIBLE_DEVICES=${GPU_ID} python main.py -m SMORE -d ${dataset} ${extra_args} > "../${LOG_FILE}" 2>&1
    EXIT_CODE=$?
    cd ..
    if [ ${EXIT_CODE} -eq 0 ]; then
        echo "[${COUNT}/${TOTAL}] ${dataset} | ${exp_name} | DONE"
    else
        echo "[${COUNT}/${TOTAL}] ${dataset} | ${exp_name} | FAILED (exit ${EXIT_CODE})"
    fi
done

echo "============================================================"
echo " All ${TOTAL} re-runs finished!"
echo " grep 'Test:' ${LOG_DIR}/*.log"
echo "============================================================"
