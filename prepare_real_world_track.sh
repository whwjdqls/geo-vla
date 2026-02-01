#!/bin/bash
# Real World 데이터 준비 스크립트 - Track Head 학습용
#
# 사용법:
#   ./prepare_real_world_track.sh <dataset_name>
#
# 예시:
#   ./prepare_real_world_track.sh omy_f3m_pick_hat_depth_0128
#
# 입력:
#   - 원본 데이터: /weka/jisookim/dataset/real_world/<dataset_name>
#   - Track 데이터: /weka/jisookim/dataset/real_world/<dataset_name>_hdf5/real_world/<dataset_name>/pointrack/results
#
# 출력:
#   - 변환된 데이터: /weka/jisookim/dataset/real_world_lerobot/<dataset_name>_pt

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <dataset_name>"
    echo "Example: $0 omy_f3m_pick_hat_depth_0128"
    exit 1
fi

DATASET_NAME=$1
BASE_DIR="/weka/jisookim/dataset/real_world"

INPUT_DIR="${BASE_DIR}/${DATASET_NAME}"
TRACK_DIR="${BASE_DIR}/${DATASET_NAME}_hdf5/real_world/${DATASET_NAME}/pointrack/results"
OUTPUT_DIR="/weka/jisookim/dataset/real_world_lerobot/${DATASET_NAME}_pt"

echo "============================================"
echo "Preparing Track Data for: ${DATASET_NAME}"
echo "============================================"
echo "Input:  ${INPUT_DIR}"
echo "Track:  ${TRACK_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo ""

# Check if input exists
if [ ! -d "${INPUT_DIR}" ]; then
    echo "Error: Input directory not found: ${INPUT_DIR}"
    exit 1
fi

# Check if track dir exists
if [ ! -d "${TRACK_DIR}" ]; then
    echo "Error: Track directory not found: ${TRACK_DIR}"
    exit 1
fi

# Run preparation script
python scripts/prepare_real_world_data.py \
    --input_dir "${INPUT_DIR}" \
    --track_dir "${TRACK_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --mode track \
    --num_points 1024

echo ""
echo "============================================"
echo "Done! Data saved to: ${OUTPUT_DIR}"
echo ""
echo "To train track head model, use:"
echo "  ./run_pi05_newhead_openvla_full_finetune_robocasa.sh"
echo "  with --data-use_local_data --data-root_dir ${OUTPUT_DIR}"
echo "============================================"
