# Multi-GPU Training for Voxel Gaussian Diffusion

This document explains how to use the multi-GPU training capabilities of the voxel gaussian diffusion model.

## Overview

The `voxel_gaussian_train_clean.py` script now supports distributed training across multiple GPUs using MPI (Message Passing Interface). The script automatically:

- Distributes data loading across GPUs (each GPU processes different data)
- Uses PyTorch's DistributedDataParallel (DDP) for model parallelism
- Synchronizes gradients across GPUs
- Logs metrics only from the primary rank (rank 0) to avoid conflicts

## Prerequisites

1. **MPI Installation**: You need MPI installed on your system
   ```bash
   # On Ubuntu/Debian
   sudo apt-get install libopenmpi-dev openmpi-bin
   
   # On CentOS/RHEL
   sudo yum install openmpi openmpi-devel
   
   # Or using conda
   conda install mpi4py
   ```

2. **Python Dependencies**: Ensure you have all required packages
   ```bash
   pip install mpi4py torch torchvision wandb imageio
   ```

## Usage

### 1. Using the Provided Script

The easiest way to run multi-GPU training is using the provided shell script:

```bash
# Edit the script to set your paths and parameters
nano scripts/run_multi_gpu_training.sh

# Run the script
./scripts/run_multi_gpu_training.sh
```

### 2. Manual Command

You can also run the training manually with mpirun:

```bash
mpirun -np 4 python scripts/voxel_gaussian_train_clean.py \
    --data_dir /path/to/your/data \
    --norm_stats_path /path/to/normalization_stats.json \
    --batch_size 2 \
    --lr 1e-4 \
    --wandb_project "voxel-gaussian-multi-gpu" \
    --use_fp16 True \
    [other arguments...]
```

### 3. Single GPU Training

For single GPU training, simply run the script normally (without mpirun):

```bash
python scripts/voxel_gaussian_train_clean.py \
    --data_dir /path/to/your/data \
    --norm_stats_path /path/to/normalization_stats.json \
    [other arguments...]
```

## Key Parameters for Multi-GPU Training

- `--batch_size`: **Per-GPU batch size**. The effective batch size = batch_size × number_of_gpus
- `--lr`: Learning rate (you may want to scale it with the number of GPUs)
- `--use_fp16`: Recommended for faster training and reduced memory usage
- `--log_interval`: How often to log metrics (only rank 0 logs to wandb)
- `--save_interval`: How often to save checkpoints

## Performance Considerations

1. **Batch Size Scaling**: The effective batch size increases with the number of GPUs. You may need to:
   - Adjust learning rate proportionally (e.g., lr × num_gpus)
   - Ensure your data loading can keep up with multiple GPUs

2. **Memory Usage**: Each GPU loads its own data subset, so total memory usage scales with the number of GPUs

3. **Wandb Logging**: Only rank 0 performs logging and visualization to avoid conflicts and reduce overhead

## Troubleshooting

### Common Issues

1. **MPI not found**: Ensure MPI is installed and `mpirun` is in your PATH
2. **CUDA device mismatch**: The script automatically assigns GPUs using `rank % GPUS_PER_NODE`
3. **Wandb conflicts**: Only rank 0 initializes wandb, other ranks skip logging

### Debugging

To debug multi-GPU issues, you can:

1. Check if all GPUs are visible:
   ```bash
   python -c "import torch; print(f'GPUs available: {torch.cuda.device_count()}')"
   ```

2. Run with verbose MPI output:
   ```bash
   mpirun -np 4 --verbose python scripts/voxel_gaussian_train_clean.py [args...]
   ```

3. Monitor GPU usage:
   ```bash
   watch nvidia-smi
   ```

## Example Performance Comparison

| Setup | Batch Size | Effective Batch Size | Training Speed |
|-------|------------|---------------------|----------------|
| 1 GPU | 8 | 8 | 1.0x |
| 2 GPUs | 4 | 8 | ~1.8x |
| 4 GPUs | 2 | 8 | ~3.5x |
| 4 GPUs | 4 | 16 | ~3.5x (faster convergence) |

## Advanced Configuration

### Custom GPU Assignment

If you need custom GPU assignment, you can modify the `GPUS_PER_NODE` variable in `guided_diffusion/dist_util.py`.

### Learning Rate Scaling

For optimal convergence with multiple GPUs, consider scaling the learning rate:

```bash
# For 4 GPUs, scale learning rate by sqrt(4) = 2
mpirun -np 4 python scripts/voxel_gaussian_train_clean.py \
    --lr 2e-4 \  # Instead of 1e-4
    [other args...]
```

### Mixed Precision Training

Use `--use_fp16 True` for faster training and reduced memory usage:

```bash
mpirun -np 4 python scripts/voxel_gaussian_train_clean.py \
    --use_fp16 True \
    --fp16_scale_growth 1e-3 \
    [other args...]
```

This enables automatic mixed precision training which can significantly speed up training on modern GPUs. 