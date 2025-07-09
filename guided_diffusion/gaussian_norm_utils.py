"""
Clean gaussian normalization utilities for voxel diffusion training.

Based on the user's min-max normalization approach:
- opacity: min-max to [-1,1]
- scaling: log first, then min-max to [-1,1] 
- features_dc: min-max to [-1,1]
- features_rest: global min-max to [-1,1]
- rotation: normalize to unit quaternions (no scaling)
- xyz: unchanged (positional, handled separately)
"""

import json
import torch as th
import torch.nn.functional as F


def load_normalization_stats(stats_path):
    """Load global normalization statistics from JSON file."""
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    return stats


def to_logit_minmax(x, x_min, x_max):
    """
    Convert values to [-1,1] range using global min/max.
    Equivalent to user's to_logit function but with global stats.
    """
    # Guard against zero range
    x_range = max(x_max - x_min, 1e-8)
    return 2.0 * (x - x_min) / x_range - 1.0


def from_logit_minmax(z, x_min, x_max):
    """
    Inverse of to_logit_minmax: convert from [-1,1] back to original range.
    Equivalent to user's from_logit function.
    """
    return x_min + (z + 1.0) * 0.5 * (x_max - x_min)


def normalize_gaussian_volume(volume, include_features, norm_stats):
    """
    Normalize a volume tensor using global min-max statistics.
    
    Args:
        volume: [C, D, H, W] tensor with gaussian features
        include_features: List of feature names in order
        norm_stats: Dict with global min/max statistics
        
    Returns:
        Normalized volume with features in [-1,1] range
    """
    volume = volume.clone()
    channel_idx = 0
    
    for feature_name in include_features:
        if feature_name == "opacity":
            # opacity: min-max to [-1,1]
            x_min = norm_stats[f"{feature_name}_min"]
            x_max = norm_stats[f"{feature_name}_max"]
            volume[channel_idx] = to_logit_minmax(volume[channel_idx], x_min, x_max)
            channel_idx += 1
            
        elif feature_name == "scaling":
            # scaling: log first, then min-max to [-1,1]
            for i in range(3):
                ch = volume[channel_idx + i]
                # Apply log transform first
                log_ch = th.log(th.clamp(ch, min=1e-8))
                # Then normalize to [-1,1] using global log stats
                x_min = norm_stats[f"{feature_name}_log_min"]
                x_max = norm_stats[f"{feature_name}_log_max"]
                volume[channel_idx + i] = to_logit_minmax(log_ch, x_min, x_max)
            channel_idx += 3
            
        elif feature_name == "rotation":
            # rotation: normalize to unit quaternions (no scaling to [-1,1])
            quat_start = channel_idx
            quat_end = channel_idx + 4
            quat = volume[quat_start:quat_end]  # [4, D, H, W]
            quat_flat = quat.reshape(4, -1)  # [4, D*H*W]
            quat_norm = th.norm(quat_flat, dim=0, keepdim=True)
            quat_flat = quat_flat / (quat_norm + 1e-6)
            volume[quat_start:quat_end] = quat_flat.reshape_as(quat)
            channel_idx += 4
            
        elif feature_name == "features_dc":
            # features_dc: min-max to [-1,1]
            for i in range(3):
                ch = volume[channel_idx + i]
                x_min = norm_stats[f"{feature_name}_min"]
                x_max = norm_stats[f"{feature_name}_max"]
                volume[channel_idx + i] = to_logit_minmax(ch, x_min, x_max)
            channel_idx += 3
            
        elif feature_name == "features_rest":
            # features_rest: global min-max to [-1,1]
            x_min = norm_stats[f"{feature_name}_min"]
            x_max = norm_stats[f"{feature_name}_max"]
            for i in range(45):
                ch = volume[channel_idx + i]
                volume[channel_idx + i] = to_logit_minmax(ch, x_min, x_max)
            channel_idx += 45
    
    return volume


def denormalize_gaussian_volume(volume, include_features, norm_stats):
    """
    Denormalize a volume tensor using global min-max statistics.
    
    Args:
        volume: [C, D, H, W] normalized tensor in [-1,1] range
        include_features: List of feature names in order
        norm_stats: Dict with global min/max statistics
        
    Returns:
        Denormalized volume ready for rendering
    """
    volume = volume.clone()
    channel_idx = 0
    
    for feature_name in include_features:
        if feature_name == "opacity":
            # opacity: from [-1,1] back to original range
            x_min = norm_stats[f"{feature_name}_min"]
            x_max = norm_stats[f"{feature_name}_max"]
            volume[channel_idx] = from_logit_minmax(volume[channel_idx], x_min, x_max)
            # Clamp to valid opacity range [0, 1]
            volume[channel_idx] = th.clamp(volume[channel_idx], 0.0, 1.0)
            channel_idx += 1
            
        elif feature_name == "scaling":
            # scaling: from [-1,1] back to log space, then exp to get scales
            for i in range(3):
                ch = volume[channel_idx + i]
                # Denormalize from [-1,1] to log space
                x_min = norm_stats[f"{feature_name}_log_min"]
                x_max = norm_stats[f"{feature_name}_log_max"]
                log_ch = from_logit_minmax(ch, x_min, x_max)
                # Convert from log space back to scale space (this gives log scales, NOT scales)
                # The volume should contain log scales for rendering
                volume[channel_idx + i] = log_ch
                # Note: The rendering code will apply exp() to get actual scales
            channel_idx += 3
            
        elif feature_name == "rotation":
            # rotation: ensure unit quaternions (already normalized during forward pass)
            quat_start = channel_idx
            quat_end = channel_idx + 4
            quat = volume[quat_start:quat_end]  # [4, D, H, W]
            quat_flat = quat.reshape(4, -1)  # [4, D*H*W]
            quat_norm = th.norm(quat_flat, dim=0, keepdim=True)
            quat_flat = quat_flat / (quat_norm + 1e-6)
            volume[quat_start:quat_end] = quat_flat.reshape_as(quat)
            channel_idx += 4
            
        elif feature_name == "features_dc":
            # features_dc: from [-1,1] back to original range
            for i in range(3):
                ch = volume[channel_idx + i]
                x_min = norm_stats[f"{feature_name}_min"]
                x_max = norm_stats[f"{feature_name}_max"]
                volume[channel_idx + i] = from_logit_minmax(ch, x_min, x_max)
            channel_idx += 3
            
        elif feature_name == "features_rest":
            # features_rest: from [-1,1] back to original range
            x_min = norm_stats[f"{feature_name}_min"]
            x_max = norm_stats[f"{feature_name}_max"]
            for i in range(45):
                ch = volume[channel_idx + i]
                volume[channel_idx + i] = from_logit_minmax(ch, x_min, x_max)
            channel_idx += 45
    
    return volume


def gaussian_dict_to_volume(gaussians, include_features, grid_size):
    """
    Convert gaussian parameter dict to volume tensor (without normalization).
    
    Args:
        gaussians: Dict with gaussian parameters
        include_features: List of feature names to include
        grid_size: 3D grid size (assumes cubic grid)
        
    Returns:
        volume: [C, D, H, W] tensor
    """
    n_voxels = gaussians["opacity"].shape[0]
    expected_voxels = grid_size ** 3
    assert n_voxels == expected_voxels, f"Expected {expected_voxels} voxels but got {n_voxels}"
    
    # Calculate total channels
    feature_channels = 0
    for feature in include_features:
        if feature == "opacity":
            feature_channels += 1
        elif feature == "scaling":
            feature_channels += 3
        elif feature == "rotation":
            feature_channels += 4
        elif feature == "features_dc":
            feature_channels += 3
        elif feature == "features_rest":
            feature_channels += 45
    
    # Collect features in order
    feature_list = []
    
    if "opacity" in include_features:
        # [N, 1] -> keep as [N, 1]
        opacity = gaussians["opacity"]
        assert opacity.shape[1] == 1, f"Expected opacity shape [N, 1] but got {opacity.shape}"
        feature_list.append(opacity)
            
    if "scaling" in include_features:
        # [N, 3]
        scaling = gaussians["scaling"]
        assert scaling.shape[1] == 3, f"Expected scaling shape [N, 3] but got {scaling.shape}"
        feature_list.append(scaling)
            
    if "rotation" in include_features:
        # [N, 4]
        rotation = gaussians["rotation"]
        assert rotation.shape[1] == 4, f"Expected rotation shape [N, 4] but got {rotation.shape}"
        feature_list.append(rotation)
            
    if "features_dc" in include_features:
        # [N, 1, 3] -> [N, 3]
        features_dc = gaussians["features_dc"].squeeze(1)
        assert features_dc.shape[1] == 3, f"Expected features_dc shape [N, 3] but got {features_dc.shape}"
        feature_list.append(features_dc)
            
    if "features_rest" in include_features:
        # [N, 15, 3] -> [N, 45]
        features_rest = gaussians["features_rest"]
        features_rest = features_rest.reshape(n_voxels, -1)
        assert features_rest.shape[1] == 45, f"Expected features_rest shape [N, 45] but got {features_rest.shape}"
        feature_list.append(features_rest)

    # Concatenate all features: [N, total_channels]
    all_features = th.cat(feature_list, dim=1)
    
    # Reshape to volume: [total_channels, D, H, W]
    volume = all_features.transpose(0, 1)  # [total_channels, N]
    print(f"shape after transpose and before volume reshape: {volume.shape}")
    volume = volume.reshape(feature_channels, grid_size, grid_size, grid_size)
    
    return volume


def volume_to_gaussian_dict(volume, include_features, grid_size, voxel_centers, opacity_threshold=0.00):
    """
    Convert volume tensor back to gaussian parameter dict for rendering.
    
    Args:
        volume: [C, D, H, W] denormalized tensor
        include_features: List of feature names
        grid_size: 3D grid size
        voxel_centers: [D, H, W, 3] tensor with voxel center coordinates
        opacity_threshold: Minimum opacity for including gaussians
        
    Returns:
        Dict with gaussian parameters ready for rendering
    """
    channel_idx = 0
    
    # Extract opacity and create mask
    if "opacity" in include_features:
        opacity_vol = volume[channel_idx]  # [D, H, W]
        channel_idx += 1
    else:
        raise ValueError("Opacity not in include_features")
    
    # Apply sigmoid to opacity and create mask
    opacity_vol = opacity_vol.reshape(-1)
    mask = opacity_vol > opacity_threshold
    
    if mask.sum() == 0:
        raise ValueError(f"No voxels above opacity threshold {opacity_threshold}")
    
    # Get positions
    print(f"old voxel_centers shape: {voxel_centers.shape}")
    pos = voxel_centers.reshape(-1, 3)[mask]  # [N_valid, 3]
    print(f"new pos shape: {pos.shape}")
    
    # Build result dict
    result = {
        "xyz": pos,
        "opacity": opacity_vol[mask, None],  # [N_valid, 1]
    }
    
    # Extract other features
    if "scaling" in include_features:
            # Extract log scales and convert to scales with exp
            scaling_vol = volume[channel_idx:channel_idx+3]  # [3, D, H, W]
            scaling_flat = scaling_vol.permute(1, 2, 3, 0).reshape(-1, 3)[mask]  # [N_valid, 3]
            
            # Debug: Check for problematic values before exp
            if th.any(th.isnan(scaling_flat)) or th.any(th.isinf(scaling_flat)):
                raise ValueError(f"WARNING: NaN/Inf in log scaling values - replacing with safe defaults")
            
            result["scaling"] = scaling_flat
            channel_idx += 3
            
    if "rotation" in include_features:
            # Extract and normalize quaternions
            rotation_vol = volume[channel_idx:channel_idx+4]  # [4, D, H, W]
            rotation_flat = rotation_vol.permute(1, 2, 3, 0).reshape(-1, 4)[mask]  # [N_valid, 4]
            result["rotation"] = rotation_flat
            channel_idx += 4
            
    if "features_dc" in include_features:
            # Extract color features
            dc_vol = volume[channel_idx:channel_idx+3]  # [3, D, H, W]
            dc_flat = dc_vol.permute(1, 2, 3, 0).reshape(-1, 3)[mask]  # [N_valid, 3]
            result["features_dc"] = dc_flat[:, None, :]  # [N_valid, 1, 3]
            channel_idx += 3
            
    if "features_rest" in include_features:
            # Extract SH features
            rest_vol = volume[channel_idx:channel_idx+45]  # [45, D, H, W]
            rest_flat = rest_vol.permute(1, 2, 3, 0).reshape(-1, 45)[mask]  # [N_valid, 45]
            result["features_rest"] = rest_flat.reshape(-1, 15, 3)  # [N_valid, 15, 3]
            channel_idx += 45
    
    return result 