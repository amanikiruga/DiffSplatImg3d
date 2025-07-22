#!/bin/bash

# Launch script for plenoxel diffusion training
# Usage: ./launch_plenoxel_training.sh [num_gpus]

set -e

# Configuration
DATA_DIR="/weka/scratch/weka/tenenbaum/akiruga/svox2/data/ckpts/shapenet_chairs_all_jupyter"
NORM_STATS_PATH="./norm_stats.pt"
PROJECT_NAME="plenoxels-diffusion-$(date +%Y%m%d)"
RUN_NAME="plenoxel-training-$(date +%H%M%S)"

# Training parameters
BATCH_SIZE=8
LEARNING_RATE=1e-4
MAX_STEPS=100000  # For testing, increase for full training
LOG_INTERVAL=50
SAVE_INTERVAL=5000
TEST_GEN_INTERVAL=1000

# Model parameters
VOLUME_SIZE=32
DIFFUSION_STEPS=1000
NOISE_SCHEDULE="linear"

# Get number of GPUs (default to 1)
NUM_GPUS=${1:-1}

echo "=== PLENOXEL DIFFUSION TRAINING LAUNCHER ==="
echo "Data directory: $DATA_DIR"
echo "Normalization stats: $NORM_STATS_PATH"
echo "Number of GPUs: $NUM_GPUS"
echo "Batch size: $BATCH_SIZE (effective: $((BATCH_SIZE * NUM_GPUS)))"
echo "Max steps: $MAX_STEPS"
echo "Project: $PROJECT_NAME"
echo "Run: $RUN_NAME"
echo ""

# Check if normalization stats exist
if [ ! -f "$NORM_STATS_PATH" ]; then
    echo "⚠️  Normalization stats not found at $NORM_STATS_PATH"
    echo "Generating normalization statistics..."
    python scripts/generate_plenoxel_norm_stats.py \
        --data_dir "$DATA_DIR" \
        --output "$NORM_STATS_PATH" \
        --validate
    echo "✓ Normalization stats generated"
    echo ""
fi

# Set environment variables
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
export NCCL_DEBUG=INFO
export PYTHONPATH="$PWD:$PYTHONPATH"

# Prepare training command
TRAIN_CMD="python scripts/voxel_gaussian_train_plenoxels.py \
    --data_dir '$DATA_DIR' \
    --norm_stats_path '$NORM_STATS_PATH' \
    --wandb_project '$PROJECT_NAME' \
    --wandb_run '$RUN_NAME' \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --max_steps $MAX_STEPS \
    --volume_size $VOLUME_SIZE \
    --diffusion_steps $DIFFUSION_STEPS \
    --noise_schedule '$NOISE_SCHEDULE' \
    --log_interval $LOG_INTERVAL \
    --save_interval $SAVE_INTERVAL \
    --test_generation_interval $TEST_GEN_INTERVAL \
    --use_checkpoint \
    --use_scale_shift_norm \
    --random_flip \
    --random_rotate \
    --num_workers 4 \
    --seed 42"

echo "Starting training..."
echo "Command: $TRAIN_CMD"
echo ""

# Launch training
if [ $NUM_GPUS -eq 1 ]; then
    # Single GPU training
    echo "🚀 Single GPU training"
    eval $TRAIN_CMD
else
    # Multi-GPU training with MPI
    echo "🚀 Multi-GPU training ($NUM_GPUS GPUs)"
    mpiexec -n $NUM_GPUS --allow-run-as-root $TRAIN_CMD
fi

echo ""
echo "✓ Training completed!"
echo "Check wandb for logs: https://wandb.ai"
echo "Checkpoints saved in: ./checkpoints-plenoxels-diffusion/" 