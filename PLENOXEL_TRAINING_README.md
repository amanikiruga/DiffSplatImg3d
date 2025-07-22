# Plenoxel 3D Diffusion Training

This directory contains a complete pipeline for training 3D diffusion models on plenoxel data extracted from svox2 optimization. The system uses dense 32×32×32 voxel grids with 28 channels (1 density + 27 spherical harmonics) and applies the exact normalization and rendering techniques from the svox2 optimization notebook.

## Overview

The plenoxel diffusion training system consists of:

1. **Data Processing**: Loads dense_grid.npz files from svox2 optimization
2. **Normalization**: Uses log1p + z-score normalization for density, z-score for SH
3. **3D Diffusion Model**: UNet3D architecture for unconditional generation
4. **Rendering**: Converts generated plenoxels back to svox2 format for visualization
5. **Extensive Logging**: WandB integration with rendered videos and images

## Quick Start

### 1. Generate Normalization Statistics

First, compute normalization statistics from your plenoxel data:

```bash
python scripts/generate_plenoxel_norm_stats.py \
    --data_dir "/weka/scratch/weka/tenenbaum/akiruga/svox2/data/ckpts/shapenet_chairs_all_jupyter" \
    --output "norm_stats.pt" \
    --validate
```

### 2. Launch Training

Use the provided launch script for easy training:

```bash
# Single GPU training
./scripts/launch_plenoxel_training.sh 1

# Multi-GPU training (4 GPUs)
./scripts/launch_plenoxel_training.sh 4
```

### 3. Manual Training

For more control, run the training script directly:

```bash
python scripts/voxel_gaussian_train_plenoxels.py \
    --data_dir "/weka/scratch/weka/tenenbaum/akiruga/svox2/data/ckpts/shapenet_chairs_all_jupyter" \
    --norm_stats_path "norm_stats.pt" \
    --wandb_project "plenoxels-diffusion" \
    --wandb_run "my-experiment" \
    --batch_size 8 \
    --lr 1e-4 \
    --max_steps 100000 \
    --volume_size 32 \
    --use_checkpoint \
    --use_scale_shift_norm \
    # --random_flip \
    # --random_rotate
```

## Data Format

The system expects dense_grid.npz files with the following structure:

```python
# Each file contains:
data = np.load("dense_grid.npz")
dense_grid = data["dense_grid"]  # Shape: (32, 32, 32, 28)

# Channel layout:
# - Channel 0: Density
# - Channels 1-27: Spherical harmonics coefficients (9 SH bases × 3 RGB)
```

## Normalization Pipeline

The normalization follows the exact approach from opt.ipynb:

1. **Density Channel (0)**: Apply `log1p(density.clamp_min(0) + 1e-6)`
2. **All Channels**: Z-score normalization `(x - μ) / σ`
3. **Scaling**: Divide by channel-wise 99.9th percentile to get [-1,1] range
4. **Safety Clamp**: `clamp(-1, 1)`

For denormalization, the process is reversed:
1. Multiply by amax values
2. Inverse z-score: `x * σ + μ`
3. For density: `torch.expm1(x)` to inverse the log1p

## Model Architecture

- **Base Model**: 3D U-Net with skip connections
- **Input/Output**: 28 channels, 32×32×32 resolution
- **Training**: Unconditional diffusion with custom α²_t loss weighting
- **Multi-GPU**: Distributed training support via MPI

## Rendering and Visualization

The system includes comprehensive rendering capabilities:

1. **Denormalization**: Converts normalized [-1,1] tensors back to original scale
2. **svox2 Conversion**: Creates SparseGrid objects for rendering
3. **Video Generation**: Multi-view orbital videos (60 camera poses)
4. **Image Grids**: Multi-view image grids for easy comparison

### Logged Visualizations

During training, the following are logged to WandB:

- **Ground Truth**: Original training data rendered as videos/images
- **Noisy Input**: Visualization of noisy training inputs
- **Model Output**: Denoised predictions during training
- **Test Generation**: Full denoising from pure noise (periodic)

## Configuration

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_dir` | Required | Directory containing dense_grid.npz files |
| `--norm_stats_path` | Required | Path to normalization statistics |
| `--batch_size` | 8 | Training batch size |
| `--lr` | 1e-4 | Learning rate |
| `--volume_size` | 32 | Voxel grid resolution |
| `--diffusion_steps` | 1000 | Number of diffusion timesteps |
| `--max_steps` | None | Maximum training steps |
| `--log_interval` | 10 | Logging frequency |
| `--save_interval` | 10000 | Checkpoint save frequency |
| `--test_generation_interval` | 500 | Test generation frequency |

### Data Augmentation

- `--random_flip`: Random axis flipping
- `--random_rotate`: Random 90° rotations in XY plane

### Model Options

- `--use_checkpoint`: Gradient checkpointing for memory efficiency
- `--use_scale_shift_norm`: Scale-shift normalization in ResNet blocks
- `--dropout`: Dropout rate (default: 0.0)
- `--learn_sigma`: Learn noise variance (doubles output channels)

## Directory Structure

```
├── scripts/
│   ├── voxel_gaussian_train_plenoxels.py    # Main training script
│   ├── generate_plenoxel_norm_stats.py      # Normalization statistics
│   └── launch_plenoxel_training.sh          # Convenient launcher
├── guided_diffusion/
│   ├── script_util_3d.py                    # 3D model utilities
│   └── unet_3d.py                           # 3D U-Net architecture
├── checkpoints-plenoxels-diffusion/         # Training outputs
│   ├── debug_images/                        # Local debug images
│   └── model_*.pt                           # Model checkpoints
└── norm_stats.pt                            # Normalization statistics
```

## Monitoring

### WandB Integration

All training metrics and visualizations are logged to WandB:

- **Metrics**: Loss, MSE, learning rate, training steps
- **Images**: Multi-view grids of rendered plenoxels
- **Videos**: Orbital camera trajectories around generated objects
- **Debug**: Local saves of all visualizations for offline inspection

### Local Debug Output

Debug images and videos are saved locally in:
- `checkpoints-plenoxels-diffusion/*/debug_images/`
- Individual frames: `*_frame_*.png`
- Multi-view grids: `*_grid_step_*.png`
- Videos: `*_video_step_*.mp4`

## Dependencies

### Required Packages

- PyTorch >= 1.9
- numpy
- imageio
- wandb
- tqdm
- mpi4py (for multi-GPU training)

### External Dependencies

- **svox2**: For plenoxel rendering (optional, falls back to dummy images)
- **splatter-image**: For camera pose utilities (optional)

### Environment Setup

```bash
export SPLATTER_IMAGE_ROOT="/om/user/akiruga/splatter-image"
export SVOX2_ROOT="/om/user/akiruga/svox2"
export PYTHONPATH="$PWD:$PYTHONPATH"
```

## Troubleshooting

### Common Issues

1. **Import Error: svox2 not found**
   - The system will fallback to dummy rendering
   - Install svox2 or set `SVOX2_ROOT` environment variable

2. **CUDA Out of Memory**
   - Reduce `--batch_size`
   - Enable `--use_checkpoint`
   - Reduce model parameters

3. **MPI Training Issues**
   - Ensure `mpi4py` is installed
   - Check `CUDA_VISIBLE_DEVICES` setting
   - Use `--allow-run-as-root` if needed

4. **No dense_grid.npz files found**
   - Check the `--data_dir` path
   - Ensure files are in subdirectories (recursive search)

### Performance Tips

- Use `--use_checkpoint` for memory efficiency
- Enable data augmentation (`--random_flip`, `--random_rotate`)
- Adjust `--num_workers` for data loading
- Use multi-GPU training for faster convergence

## Citation

If you use this code, please cite the relevant papers:

```bibtex
@article{svox2,
  title={Plenoxels: Radiance Fields without Neural Networks},
  author={Yu, Alex and Fridovich-Keil, Sara and Tancik, Matthew and Chen, Qinhong and Recht, Benjamin and Kanazawa, Angjoo},
  journal={CVPR},
  year={2022}
}

@article{diffusion,
  title={Denoising Diffusion Probabilistic Models},
  author={Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  journal={NeurIPS},
  year={2020}
}
``` 