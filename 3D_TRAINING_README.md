# 3D Diffusion Model Training

This directory contains the implementation for training 3D diffusion models on volumetric data, following the specifications provided for neural field training with specialized parameters.

## Overview

The 3D training system includes:
- **3D U-Net Architecture**: 4 scaling blocks with 2 ResNet blocks per scale
- **Channel Progression**: Linear increase from 64 to 256 channels  
- **Skip Attention**: At scaling factors 2, 4, and 8 with 32 channels per head
- **Custom Beta Scheduling**: Linear from 0.0015 to 0.05 at 1000 timesteps
- **Specialized Loss Weighting**: ωt = ᾱ²t (alpha bar squared)

## Training Specifications

The implementation follows these training details:
- **Batch Size**: 8
- **Optimizer**: Adam with initial learning rate of 10⁻⁴
- **Learning Rate Schedule**: Decaying from 10⁻⁴ to 10⁻⁶ over 3M iterations
- **Voxel Grid Resolution**: 32³
- **Target Training**: 2 GPUs (multi-GPU support via DDP)

## File Structure

```
guided_diffusion/
├── volume_datasets.py          # 3D volumetric data loading
├── script_util_3d.py          # 3D model creation utilities
└── unet_3d.py                 # 3D U-Net implementation

scripts/
├── volume_train.py            # Main 3D training script
└── train_3d_example.py        # Example usage script

example_3d_unet.py             # Model demonstration
```

## Quick Start

### 1. Prepare Your Data

Your volumetric data should be stored as `.npy` or `.npz` files in a directory structure like:

```
data/
└── volumes/
    ├── volume_001.npy
    ├── volume_002.npy
    └── ...
```

Each volume file should contain 3D data in one of these formats:
- **Shape (D, H, W)**: Single channel volume
- **Shape (C, D, H, W)**: Multi-channel volume 
- **Shape (D, H, W, C)**: Multi-channel volume (will be transposed)

### 2. Create Sample Data (for testing)

```bash
python scripts/train_3d_example.py --create_sample_data
```

### 3. Run Training

#### Basic Training
```bash
python scripts/volume_train.py \
    --data_dir ./data/volumes \
    --batch_size 8 \
    --volume_size 32 \
    --lr 1e-4 \
    --lr_anneal_steps 3000000
```

#### Using the Example Script
```bash
python scripts/train_3d_example.py
```

#### Multi-GPU Training (2 GPUs)
```bash
mpiexec -n 2 python scripts/volume_train.py \
    --data_dir ./data/volumes \
    --batch_size 8 \
    --volume_size 32 \
    --lr 1e-4 \
    --lr_anneal_steps 3000000
```

## Key Parameters

### Model Architecture
- `--volume_size 32`: Voxel grid resolution (32³)
- `--in_channels 3`: Input channels
- `--out_channels 3`: Output channels

### Training Configuration
- `--batch_size 8`: Batch size
- `--lr 1e-4`: Initial learning rate (10⁻⁴)
- `--lr_anneal_steps 3000000`: Steps for LR decay (3M iterations)
- `--weight_decay 0.0`: Weight decay

### Custom Diffusion Parameters
- `--beta_start 0.0015`: Beta schedule start value
- `--beta_end 0.05`: Beta schedule end value
- `--diffusion_steps 1000`: Number of diffusion timesteps

### Logging and Checkpointing
- `--log_interval 100`: Log every N steps
- `--save_interval 10000`: Save checkpoint every N steps
- `--resume_checkpoint path/to/checkpoint.pt`: Resume from checkpoint

## Advanced Usage

### Custom Data Loading

For specialized data formats, modify `guided_diffusion/volume_datasets.py`:

```python
class CustomVolumeDataset(VolumeDataset):
    def __getitem__(self, idx):
        # Your custom data loading logic
        volume = your_custom_loader(self.local_volumes[idx])
        return volume, {}
```

### Custom Loss Functions

Extend the `Custom3DTrainLoop` in `scripts/volume_train.py`:

```python
class YourCustomTrainLoop(Custom3DTrainLoop):
    def forward_backward(self, batch, cond):
        # Your custom loss computation
        pass
```

### Rendering Integration

For neural radiance field (NeRF) style training with:
- 4 random training views at 128×128 resolution
- 8192 random pixels for rendering supervision  
- 92 z-steps for volumetric rendering

You would need to integrate a rendering module. Contact the development team for NeRF-specific extensions.

## Model Architecture Details

The 3D U-Net follows this architecture:

```
Input (3, 32, 32, 32)
├── Input Block: Conv3D(3→64)
├── Level 0: 2×ResBlock3D(64), Attention(64) at ds=2
├── Downsample: Conv3D(64→128, stride=2)
├── Level 1: 2×ResBlock3D(128), Attention(128) at ds=4  
├── Downsample: Conv3D(128→192, stride=2)
├── Level 2: 2×ResBlock3D(192), Attention(192) at ds=8
├── Downsample: Conv3D(192→256, stride=2)
├── Level 3: 2×ResBlock3D(256)
├── Middle: ResBlock3D(256) → Attention(256) → ResBlock3D(256)
├── Upsample + Skip Connections (reverse order)
└── Output: Conv3D(64→3)
```

**Key Features:**
- **Channel Progression**: [64, 128, 192, 256] (linear increase)
- **Attention**: Only at scaling factors 2, 4, 8
- **Attention Heads**: 32 channels per head
- **ResNet Blocks**: 2 per scaling level

## Loss Function

The training uses a custom loss weighting scheme:

```
Loss = MSE(target, prediction) × ωt
where ωt = ᾱ²t (alpha_bar_t squared)
```

This weighting emphasizes different noise levels according to the cumulative alpha schedule.

## Performance Tips

1. **Memory Optimization**:
   - Use `--use_fp16 true` for mixed precision training
   - Adjust `--microbatch` if running out of memory

2. **Multi-GPU Training**:
   - Use `mpiexec -n 2` for 2 GPU training
   - Ensure each GPU has sufficient memory for batch_size/n_gpus

3. **Data Loading**:
   - Use SSD storage for faster data loading
   - Consider data preprocessing for optimal formats

## Troubleshooting

### Common Issues

1. **Out of Memory**:
   ```bash
   # Reduce batch size or use microbatching
   --batch_size 4 --microbatch 2
   ```

2. **Data Format Errors**:
   ```python
   # Check your volume shapes
   volume = np.load("your_volume.npy")
   print(f"Volume shape: {volume.shape}")
   ```

3. **Import Errors**:
   ```bash
   # Ensure you're in the project root directory
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   ```

### Monitoring Training

Monitor key metrics:
- `train_loss`: Overall training loss
- `train_mse`: MSE component before weighting
- Learning rate decay progress
- GPU memory usage

## Citation

If you use this 3D diffusion training code, please cite the original guided-diffusion work and any relevant papers for your specific application domain.

## Support

For questions about the 3D training implementation:
1. Check the existing 2D `image_train.py` for reference patterns
2. Review the `example_3d_unet.py` for model architecture details
3. Examine `guided_diffusion/unet_3d.py` for model implementation 