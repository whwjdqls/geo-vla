#!/bin/bash
# Base Model 학습 스크립트 - Real World 데이터용 (point cloud 없음)
#
# 사용법:
#   ./run_base_real_world.sh <dataset_name> [gpu_ids] [num_train_steps]
#
# 예시:
#   ./run_base_real_world.sh omy_f3m_pick_hat_depth_0128_base 0
#   ./run_base_real_world.sh omy_f3m_pick_hat_depth_0128_base 0,1,2,3
#   ./run_base_real_world.sh omy_f3m_pick_hat_depth_0128_base 0,1,2,3 50000

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <dataset_name> [gpu_ids] [num_train_steps]"
    echo "Example: $0 omy_f3m_pick_hat_depth_0128_base 0"
    echo "Example: $0 omy_f3m_pick_hat_depth_0128_base 0,1,2,3 50000"
    exit 1
fi

DATASET_NAME=$1
GPU_IDS=${2:-0}
NUM_TRAIN_STEPS=${3:-30000}

# Count GPUs
IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS=${#GPU_ARRAY[@]}

DATA_DIR="/weka/jisookim/dataset/real_world_lerobot/${DATASET_NAME}"
EXP_NAME="pi05_base_${DATASET_NAME}"

echo "============================================"
echo "Training Base Model (Real World)"
echo "============================================"
echo "Dataset: ${DATASET_NAME}"
echo "Data Dir: ${DATA_DIR}"
echo "GPUs: ${GPU_IDS} (${NUM_GPUS} GPUs)"
echo "Train Steps: ${NUM_TRAIN_STEPS}"
echo "Exp Name: ${EXP_NAME}"
echo ""

# Check if data exists
if [ ! -d "${DATA_DIR}" ]; then
    echo "Error: Data directory not found: ${DATA_DIR}"
    echo "Please run prepare_real_world_data.py with --mode base first."
    exit 1
fi

export CUDA_VISIBLE_DEVICES=${GPU_IDS}

torchrun --standalone --nnodes=1 --nproc_per_node=${NUM_GPUS} scripts/train_pytorch.py \
    pi05_real_world_base \
    --exp_name ${EXP_NAME} \
    --batch-size 128 \
    --no-wandb-enabled \
    --save-interval 5000 \
    --num-train-steps ${NUM_TRAIN_STEPS} \
    --data.root_dir ${DATA_DIR}
