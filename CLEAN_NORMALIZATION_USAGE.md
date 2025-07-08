# Clean Gaussian Voxel Normalization System

This document describes the new clean min-max normalization system that replaces the complex percentile-based normalization. The new approach is based on the user's notebook example and provides simple, invertible normalization.

## Overview

The new system consists of:

1. **Global statistics computation** - computes min/max values across entire dataset
2. **Clean normalization utilities** - simple min-max mapping to [-1,1]
3. **Clean dataset implementation** - uses pre-computed global stats
4. **Clean training script** - no complex normalization code

## Normalization Strategy

Following the user's notebook approach:

- **Opacity**: `min-max → [-1,1]`
- **Scaling**: `log first → min-max → [-1,1]`
- **Features DC**: `min-max → [-1,1]` 
- **Features Rest**: `global min-max → [-1,1]`
- **Rotation**: `normalize to unit quaternions (no scaling)`
- **XYZ**: `unchanged (positional)`

## Usage Workflow

### Step 1: Compute Global Statistics

First, run the statistics computation script on your entire dataset:

```bash
python scripts/compute_global_norm_stats.py \
    --data_dir /path/to/your/voxel/data \
    --output_path global_norm_stats.json \
    --include_features opacity scaling rotation features_dc features_rest
```

This will:
- Scan all `gaussians.pt` files in the data directory
- Compute global min/max values for each feature type
- Save statistics to JSON file

**Example Output:**
```json
{
  "opacity_min": 0.0001,
  "opacity_max": 0.9999,
  "scaling_log_min": -4.6052,
  "scaling_log_max": 2.3026,
  "features_dc_min": -2.1234,
  "features_dc_max": 1.8765,
  "features_rest_min": -1.5432,
  "features_rest_max": 1.2345
}
```

### Step 2: Train with Clean Normalization

Use the new clean training script:

```bash
python scripts/voxel_gaussian_train_clean.py \
    --data_dir /path/to/your/voxel/data \
    --norm_stats_path global_norm_stats.json \
    --batch_size 8 \
    --lr 1e-4 \
    --wandb_project voxel-gaussian-clean \
    --wandb_run my_clean_experiment \
    --max_steps 1000
```

Required arguments:
- `--norm_stats_path`: Path to the JSON file from Step 1
- `--data_dir`: Directory containing your voxel gaussian data

## Key Files

### New Files Created:
1. **`scripts/compute_global_norm_stats.py`** - Computes global min/max statistics
2. **`guided_diffusion/gaussian_norm_utils.py`** - Clean normalization utilities
3. **`guided_diffusion/voxel_gaussian_datasets_clean.py`** - Clean dataset implementation
4. **`scripts/voxel_gaussian_train_clean.py`** - Clean training script

### Key Functions:

**Normalization:**
- `normalize_gaussian_volume()` - Apply global min-max normalization
- `denormalize_gaussian_volume()` - Reverse normalization for rendering
- `to_logit_minmax() / from_logit_minmax()` - Core min-max mapping functions

**Data Loading:**
- `load_clean_voxel_gaussian_data()` - Clean data loader with pre-computed stats
- `CleanVoxelGaussianDataset` - Dataset class using global normalization

## Benefits of New System

1. **Simplicity**: Clear min-max mapping instead of percentile-based stats
2. **Consistency**: Global statistics ensure consistent normalization across train/test
3. **Invertibility**: Perfect reconstruction using exact inverse functions
4. **Transparency**: Easy to understand and debug normalization process
5. **Match Paper**: Follows the exact approach from the user's notebook

## Training Parameters

The clean training script maintains all the paper's training parameters:

- ✅ Batch size of 8
- ✅ Adam optimizer with lr=1e-4  
- ✅ Linear beta scheduling from 0.0015 to 0.05
- ✅ 1000 diffusion timesteps
- ✅ Custom loss weighting with ωt = ᾱ²t
- ✅ 3M iterations with LR decay
- ✅ Voxel grid resolution of 32³

## Migration from Old System

The old complex normalization system in `voxel_gaussian_datasets.py` is replaced by the clean system. To migrate:

1. Compute global stats with `compute_global_norm_stats.py`
2. Use `voxel_gaussian_train_clean.py` instead of `voxel_gaussian_train_wandb.py`
3. Update any custom code to use the new normalization utilities

## Troubleshooting

**Q: Training fails with "Normalization stats file not found"**
A: Make sure to run Step 1 first to generate the global statistics JSON file.

**Q: Rendering produces strange results**
A: Check that the normalization stats were computed on the same dataset you're training on.

**Q: Memory issues during stats computation**
A: The script automatically subsamples large tensors. You can modify the subsampling size in `compute_global_min_max_stats()`.

This new system provides a much cleaner and more maintainable approach to normalization while maintaining perfect compatibility with the existing training pipeline. 