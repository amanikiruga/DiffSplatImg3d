#!/bin/bash

# Multi-GPU training script for voxel gaussian diffusion
# This script uses mpirun to launch training on multiple GPUs

# Configuration
NUM_GPUS=4  # Using 4 GPUs
DATA_DIR="/path/to/your/voxel/data"  # Update this to your actual data path
NORM_STATS_PATH="/path/to/your/normalization_stats.json"  # Update this to your actual stats file

# Training parameters
BATCH_SIZE=2  # Per GPU batch size (effective batch size = BATCH_SIZE * NUM_GPUS)
LEARNING_RATE=1e-4
MAX_STEPS=100000
LOG_INTERVAL=10
SAVE_INTERVAL=5000

# WandB configuration
WANDB_PROJECT="voxel-gaussian-diffusion-clean"
WANDB_RUN="multi_gpu_experiment_$(date +%Y%m%d_%H%M%S)"

echo "Starting multi-GPU training with $NUM_GPUS GPUs"
echo "Effective batch size: $((BATCH_SIZE * NUM_GPUS))"
echo "Data directory: $DATA_DIR"
echo "Normalization stats: $NORM_STATS_PATH"

# Run training with MPI
mpirun -np $NUM_GPUS python scripts/voxel_gaussian_train_clean.py \
    --data_dir "$DATA_DIR" \
    --norm_stats_path "$NORM_STATS_PATH" \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --max_steps $MAX_STEPS \
    --log_interval $LOG_INTERVAL \
    --save_interval $SAVE_INTERVAL \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run "$WANDB_RUN" \
    --use_fp16 True \
    --log_test_generation True \
    --test_generation_interval 1000 \
    --include_features opacity scaling rotation features_dc features_rest \
    --volume_size 32 \
    --diffusion_steps 1000 \
    --noise_schedule linear \
    --beta_start 0.0015 \
    --beta_end 0.05

echo "Training completed!" 