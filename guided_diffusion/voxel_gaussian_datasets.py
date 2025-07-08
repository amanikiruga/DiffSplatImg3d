import math
import random
import os
import gzip
import json
from pathlib import Path

import blobfile as bf
import numpy as np
import torch as th
from torch.utils.data import DataLoader, Dataset

# Make MPI import optional for testing
try:
    from mpi4py import MPI
    HAS_MPI = True
except ImportError:
    HAS_MPI = False
    # Create a mock MPI object for testing
    class MockMPI:
        class COMM_WORLD:
            @staticmethod
            def Get_rank():
                return 0
            @staticmethod
            def Get_size():
                return 1
    MPI = MockMPI()


def load_voxel_gaussian_data(
    *,
    data_dir,
    batch_size,
    grid_size=32,
    class_cond=False,
    deterministic=False,
    random_flip=True,
    random_rotate=True,
    include_features=("opacity", "scaling", "rotation", "features_dc", "features_rest"),
):
    """
    For a voxel gaussian dataset, create a generator over (volumes, kwargs) pairs.
    
    Each volume is an NCDHW float tensor representing the gaussian parameters
    arranged in a 3D grid, and the kwargs dict contains zero or more keys.
    
    :param data_dir: dataset directory containing voxel gaussian object folders.
    :param batch_size: the batch size of each returned pair.
    :param grid_size: the size of the voxel grid (assuming cubic).
    :param class_cond: if True, include a "y" key in returned dicts for class label.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_flip: if True, randomly flip the volumes for augmentation.
    :param random_rotate: if True, randomly rotate the volumes for augmentation.
    :param include_features: tuple of feature names to include in training.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    
    all_object_dirs = _list_voxel_gaussian_dirs_recursively(data_dir)
    classes = None
    if class_cond:
        # Extract class info from meta.json.gz files
        class_names = []
        for obj_dir in all_object_dirs:
            meta_path = os.path.join(obj_dir, "meta.json.gz")
            if os.path.exists(meta_path):
                with gzip.open(meta_path, "rb") as f:
                    meta = json.loads(f.read().decode())
                    class_names.append(meta.get("category", "unknown"))
            else:
                class_names.append("unknown")
        
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    
    dataset = VoxelGaussianDataset(
        grid_size,
        all_object_dirs,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_flip=random_flip,
        random_rotate=random_rotate,
        include_features=include_features,
    )
    
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_voxel_gaussian_dirs_recursively(data_dir):
    """List all object directories containing gaussians.pt files."""
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        if bf.isdir(full_path):
            # Check if this directory contains gaussians.pt
            gaussians_path = bf.join(full_path, "gaussians.pt")
            if bf.exists(gaussians_path):
                results.append(full_path)
            else:
                # Recursively search subdirectories
                results.extend(_list_voxel_gaussian_dirs_recursively(full_path))
    return results


class VoxelGaussianDataset(Dataset):
    """
    Dataset for loading voxel gaussian data and converting to volumetric format.
    
    This dataset loads the gaussian parameters from each object directory,
    converts them to a grid format, and arranges the features as volume channels.
    """
    
    def __init__(
        self,
        grid_size,
        object_dirs,
        classes=None,
        shard=0,
        num_shards=1,
        random_flip=True,
        random_rotate=True,
        include_features=("opacity", "scaling", "rotation", "features_dc", "features_rest"),
    ):
        super().__init__()
        self.grid_size = grid_size
        self.local_object_dirs = object_dirs[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_flip = random_flip
        self.random_rotate = random_rotate
        self.include_features = include_features
        
        # Calculate total channels based on included features
        self.feature_channels = self._calculate_feature_channels()
        
        # Compute global normalization statistics
        self.global_norm_stats = self._compute_global_normalization_stats()
        
        print(f"VoxelGaussianDataset: {len(self.local_object_dirs)} objects, "
              f"{self.feature_channels} channels, features: {include_features}")
    
    def _compute_global_normalization_stats(self):
        """
        Compute global normalization statistics from a subset of the dataset.
        This ensures consistent denormalization across all samples.
        """
        print("Computing global normalization statistics...")
        
        # Sample a smaller subset of objects for computing stats (for efficiency)
        sample_size = min(10, len(self.local_object_dirs))
        sample_indices = np.linspace(0, len(self.local_object_dirs)-1, sample_size, dtype=int)
        
        all_features = {name: [] for name in self.include_features}
        
        for idx in sample_indices:
            obj_dir = self.local_object_dirs[idx]
            gaussians_path = os.path.join(obj_dir, "gaussians.pt")
            
            try:
                gaussians = th.load(gaussians_path, map_location="cpu")
                
                for feature_name in self.include_features:
                    if feature_name == "opacity":
                        all_features[feature_name].append(gaussians["opacity"].float().flatten())
                    elif feature_name == "scaling":
                        all_features[feature_name].append(gaussians["scaling"].float().flatten())
                    elif feature_name == "rotation":
                        all_features[feature_name].append(gaussians["rotation"].float().flatten())
                    elif feature_name == "features_dc":
                        all_features[feature_name].append(gaussians["features_dc"].squeeze(1).float().flatten())
                    elif feature_name == "features_rest":
                        all_features[feature_name].append(gaussians["features_rest"].float().flatten())
            except Exception as e:
                print(f"Warning: Failed to load {gaussians_path}: {e}")
                continue
        
        # Compute percentile-based statistics
        stats = {}
        
        for feature_name, feature_data in all_features.items():
            if not feature_data:
                continue
                
            # Concatenate all data for this feature
            all_data = th.cat(feature_data, dim=0)
            
            # Subsample if tensor is too large (to avoid memory issues)
            if all_data.numel() > 1000000:  # If more than 1M elements
                indices = th.randperm(all_data.numel())[:100000]  # Sample 100K elements
                all_data = all_data.flatten()[indices]
            
            if feature_name == "opacity":
                # Simple linear mapping [0,1] -> [-1,1]
                stats[f'{feature_name}_offset'] = 1.0
                stats[f'{feature_name}_scale'] = 0.5
                
            elif feature_name == "scaling":
                # Per-channel percentile stats for scaling
                if all_data.dim() > 1:
                    scaling_reshaped = all_data.reshape(-1, 3)
                else:
                    # Data was subsampled, use approximate channel-wise stats
                    scaling_reshaped = all_data.reshape(-1, 3) if all_data.numel() % 3 == 0 else all_data[:-(all_data.numel()%3)].reshape(-1, 3)
                
                for i in range(3):
                    if i < scaling_reshaped.shape[1]:
                        ch_data = scaling_reshaped[:, i]
                        p05 = th.quantile(ch_data, 0.05)
                        p95 = th.quantile(ch_data, 0.95)
                        range_val = max(p95 - p05, 1e-6)
                        
                        stats[f'{feature_name}_{i}_p05'] = p05.item()
                        stats[f'{feature_name}_{i}_range'] = range_val.item()
                    else:
                        # Fallback values
                        stats[f'{feature_name}_{i}_p05'] = 0.01
                        stats[f'{feature_name}_{i}_range'] = 0.1
                    
            elif feature_name == "rotation":
                # Quaternions don't need special stats
                pass
                
            elif feature_name == "features_dc":
                # Per-channel percentile stats for features_dc
                if all_data.dim() > 1:
                    dc_reshaped = all_data.reshape(-1, 3)
                else:
                    # Data was subsampled, use approximate channel-wise stats
                    dc_reshaped = all_data.reshape(-1, 3) if all_data.numel() % 3 == 0 else all_data[:-(all_data.numel()%3)].reshape(-1, 3)
                
                for i in range(3):
                    if i < dc_reshaped.shape[1]:
                        ch_data = dc_reshaped[:, i]
                        p05 = th.quantile(ch_data, 0.05)
                        p95 = th.quantile(ch_data, 0.95)
                        range_val = max(p95 - p05, 1e-6)
                        
                        stats[f'{feature_name}_{i}_p05'] = p05.item()
                        stats[f'{feature_name}_{i}_range'] = range_val.item()
                    else:
                        # Fallback values  
                        stats[f'{feature_name}_{i}_p05'] = -1.0
                        stats[f'{feature_name}_{i}_range'] = 2.0
                    
            elif feature_name == "features_rest":
                # Global percentile stats for all features_rest channels
                p05 = th.quantile(all_data, 0.05)
                p95 = th.quantile(all_data, 0.95)
                range_val = max(p95 - p05, 1e-6)
                
                stats[f'{feature_name}_p05'] = p05.item()
                stats[f'{feature_name}_range'] = range_val.item()
        
        print(f"Computed normalization stats for {len(stats)} parameters")
        print("=== NORMALIZATION STATS DEBUG ===")
        for key, val in stats.items():
            print(f"  {key}: {val}")
        print("=== END STATS ===")
        return stats
    
    def _calculate_feature_channels(self):
        """Calculate total number of channels based on included features."""
        channels = 0
        for feature in self.include_features:
            if feature == "opacity":
                channels += 1
            elif feature == "scaling":
                channels += 3
            elif feature == "rotation":
                channels += 4
            elif feature == "features_dc":
                channels += 3  # Base color (SH degree 0)
            elif feature == "features_rest":
                channels += 45  # SH degrees 1-3: 3 + 5 + 7 = 15 coefficients * 3 colors
        return channels
    
    def __len__(self):
        return len(self.local_object_dirs)
    
    def __getitem__(self, idx):
        obj_dir = self.local_object_dirs[idx]
        gaussians_path = os.path.join(obj_dir, "gaussians.pt")
        
        # Load gaussian parameters
        gaussians = th.load(gaussians_path, map_location="cpu")
        
        # Convert gaussian parameters to volumetric format
        volume = self._gaussians_to_volume(gaussians)
        
        # Apply random augmentations
        if self.random_flip and random.random() < 0.5:
            # Random flip along one spatial axis
            axis = random.choice([1, 2, 3])  # Don't flip channel axis
            volume = th.flip(volume, dims=[axis])
        
        if self.random_rotate and random.random() < 0.3:
            # Random 90-degree rotation in one plane
            axes = random.choice([(1, 2), (1, 3), (2, 3)])
            k = random.choice([1, 2, 3])
            # PyTorch doesn't have rot90 for 3D, so we implement it manually
            volume = self._rotate_volume_90(volume, k, axes)
        
        # Normalize features to reasonable ranges for training
        volume = self._normalize_features(volume)
        
        # Ensure tensor doesn't require gradients (for DataLoader compatibility)
        volume = volume.detach()
        
        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = th.tensor(self.local_classes[idx], dtype=th.long)
        
        return volume, out_dict
    
    def _gaussians_to_volume(self, gaussians):
        """
        Convert gaussian parameters to volumetric format.
        
        Args:
            gaussians: Dict containing gaussian parameters with keys:
                - opacity: [N, 1]
                - scaling: [N, 3]  
                - rotation: [N, 4]
                - features_dc: [N, 1, 3]
                - features_rest: [N, 15, 3]
                
        Returns:
            volume: [C, D, H, W] tensor where C is total feature channels
        """
        n_voxels = gaussians["opacity"].shape[0]
        expected_voxels = self.grid_size ** 3
        
        if n_voxels != expected_voxels:
            print(f"Warning: Expected {expected_voxels} voxels but got {n_voxels}")
        
        # Collect features based on what's included
        feature_list = []
        
        for feature_name in self.include_features:
            if feature_name == "opacity":
                # Shape: [N, 1] -> keep as [N, 1]
                opacity = gaussians["opacity"].float()
                feature_list.append(opacity)
                
            elif feature_name == "scaling":
                # Shape: [N, 3] -> [N, 3]
                scaling = gaussians["scaling"].float()
                feature_list.append(scaling)
                
            elif feature_name == "rotation":
                # Shape: [N, 4] -> [N, 4] (quaternion)
                rotation = gaussians["rotation"].float()
                feature_list.append(rotation)
                
            elif feature_name == "features_dc":
                # Shape: [N, 1, 3] -> [N, 3]
                features_dc = gaussians["features_dc"].squeeze(1).float()
                feature_list.append(features_dc)
                
            elif feature_name == "features_rest":
                # Shape: [N, 15, 3] -> [N, 45] (flatten SH coefficients)
                features_rest = gaussians["features_rest"].float()
                features_rest = features_rest.reshape(n_voxels, -1)
                feature_list.append(features_rest)
        
        # Concatenate all features: [N, total_channels]
        all_features = th.cat(feature_list, dim=1)
        
        # Reshape to volume: [total_channels, D, H, W]
        volume = all_features.transpose(0, 1)  # [total_channels, N]
        volume = volume.reshape(self.feature_channels, self.grid_size, 
                               self.grid_size, self.grid_size)
        
        return volume
    
    def _rotate_volume_90(self, volume, k, axes):
        """Rotate volume by k*90 degrees in the specified plane."""
        for _ in range(k):
            volume = th.transpose(volume, axes[0], axes[1])
            volume = th.flip(volume, dims=[axes[1]])
        return volume
    
    def _normalize_features(self, volume):
        """
        Normalize features to [-1, 1] range using pre-computed global statistics.
        This ensures consistent normalization across all samples.
        """
        channel_idx = 0
        
        for feature_name in self.include_features:
            if feature_name == "opacity":
                # opacity is already in [0, 1] from sigmoid, map to [-1, 1]
                volume[channel_idx] = volume[channel_idx] * 2.0 - 1.0
                channel_idx += 1
                
            elif feature_name == "scaling":
                # Use pre-computed global percentile stats
                for i in range(3):
                    ch = volume[channel_idx + i]
                    
                    # Get global stats
                    p05 = self.global_norm_stats[f'{feature_name}_{i}_p05']
                    range_val = self.global_norm_stats[f'{feature_name}_{i}_range']
                    
                    # Normalize to [-1, 1] using global stats
                    ch_norm = 2.0 * (ch - p05) / range_val - 1.0
                    volume[channel_idx + i] = th.clamp(ch_norm, -1.0, 1.0)
                    
                channel_idx += 3
                
            elif feature_name == "rotation":
                # Quaternions should already be normalized, but ensure [-1, 1] range
                for i in range(4):
                    ch = volume[channel_idx + i]
                    volume[channel_idx + i] = th.clamp(ch, -1.0, 1.0)
                channel_idx += 4
                
            elif feature_name == "features_dc":
                # Use pre-computed global percentile stats
                for i in range(3):
                    ch = volume[channel_idx + i]
                    
                    # Get global stats
                    p05 = self.global_norm_stats[f'{feature_name}_{i}_p05']
                    range_val = self.global_norm_stats[f'{feature_name}_{i}_range']
                    
                    # Normalize using global stats
                    ch_norm = 2.0 * (ch - p05) / range_val - 1.0
                    volume[channel_idx + i] = th.clamp(ch_norm, -1.0, 1.0)
                    
                channel_idx += 3
                
            elif feature_name == "features_rest":
                # Use pre-computed global stats for all features_rest channels
                p05 = self.global_norm_stats[f'{feature_name}_p05']
                range_val = self.global_norm_stats[f'{feature_name}_range']
                
                for i in range(45):
                    ch = volume[channel_idx + i]
                    ch_norm = 2.0 * (ch - p05) / range_val - 1.0
                    volume[channel_idx + i] = th.clamp(ch_norm, -1.0, 1.0)
                    
                channel_idx += 45
        
        return volume


def denormalize_gaussian_features(volume, include_features, norm_stats=None):
    """
    Properly reverse the normalization applied in _normalize_features.
    
    Args:
        volume: [C, D, H, W] normalized tensor
        include_features: List of feature names
        norm_stats: Dict of normalization statistics (required for proper denormalization)
    
    Returns:
        Denormalized volume ready for rendering
    """
    print(f"\n=== DENORMALIZE DEBUG ===")
    print(f"Input volume shape: {volume.shape}")
    print(f"Input volume range: [{volume.min().item():.4f}, {volume.max().item():.4f}]")
    print(f"Norm stats provided: {norm_stats is not None}")
    if norm_stats:
        print(f"Norm stats keys: {list(norm_stats.keys())}")
    
    if norm_stats is None:
        print("Warning: No normalization stats provided - using fallback denormalization")
        result = _fallback_denormalize(volume, include_features)
    else:
        result = _denormalize_with_stats(volume, include_features, norm_stats)
    
    print(f"Output volume range: [{result.min().item():.4f}, {result.max().item():.4f}]")
    print("=== END DENORMALIZE DEBUG ===\n")
    return result


def _denormalize_with_stats(volume, include_features, norm_stats):
    """Denormalize using the provided normalization statistics."""
    volume = volume.clone()  # Don't modify input
    channel_idx = 0
    
    print("Using global normalization stats for denormalization")
    
    for feature_name in include_features:
        if feature_name == "opacity":
            # Map from [-1, 1] back to [0, 1] 
            before = volume[channel_idx].clone()
            volume[channel_idx] = (volume[channel_idx] + 1.0) / 2.0
            volume[channel_idx] = th.clamp(volume[channel_idx], 0.0, 1.0)
            print(f"  opacity: [{before.min().item():.4f}, {before.max().item():.4f}] -> [{volume[channel_idx].min().item():.4f}, {volume[channel_idx].max().item():.4f}]")
            channel_idx += 1
            
        elif feature_name == "scaling":
            # Reverse percentile-based normalization
            for i in range(3):
                ch = volume[channel_idx + i]
                before = ch.clone()
                
                # Get stored stats
                p05 = norm_stats[f'{feature_name}_{i}_p05']
                range_val = norm_stats[f'{feature_name}_{i}_range']
                
                # Reverse: ch_norm = 2.0 * (ch - p05) / range_val - 1.0
                ch_denorm = ((ch + 1.0) / 2.0) * range_val + p05
                ch_denorm = th.clamp(ch_denorm, 0.0, 100.0)  # Reasonable scaling range
                
                volume[channel_idx + i] = ch_denorm
                print(f"  scaling_{i}: [{before.min().item():.4f}, {before.max().item():.4f}] -> [{ch_denorm.min().item():.4f}, {ch_denorm.max().item():.4f}] (p05={p05:.4f}, range={range_val:.4f})")
                
            channel_idx += 3
            
        elif feature_name == "rotation":
            # Ensure quaternions are properly normalized
            quat_start = channel_idx
            quat_end = channel_idx + 4
            quat = volume[quat_start:quat_end]
            before_norm = th.norm(quat.view(4, -1), dim=0).mean()
            
            # Normalize quaternion to unit length
            quat_flat = quat.view(4, -1)  # [4, D*H*W]
            quat_norm = th.norm(quat_flat, dim=0, keepdim=True)
            quat_flat = quat_flat / (quat_norm + 1e-6)
            volume[quat_start:quat_end] = quat_flat.view_as(quat)
            
            after_norm = th.norm(volume[quat_start:quat_end].view(4, -1), dim=0).mean()
            print(f"  rotation: norm before={before_norm.item():.4f}, after={after_norm.item():.4f}")
            channel_idx += 4
            
        elif feature_name == "features_dc":
            # Reverse percentile-based normalization
            for i in range(3):
                ch = volume[channel_idx + i]
                before = ch.clone()
                
                # Get stored stats
                p05 = norm_stats[f'{feature_name}_{i}_p05']
                range_val = norm_stats[f'{feature_name}_{i}_range']
                
                # Reverse normalization
                ch_denorm = ((ch + 1.0) / 2.0) * range_val + p05
                volume[channel_idx + i] = ch_denorm
                print(f"  features_dc_{i}: [{before.min().item():.4f}, {before.max().item():.4f}] -> [{ch_denorm.min().item():.4f}, {ch_denorm.max().item():.4f}] (p05={p05:.4f}, range={range_val:.4f})")
                
            channel_idx += 3
            
        elif feature_name == "features_rest":
            # Reverse global percentile-based normalization
            p05 = norm_stats[f'{feature_name}_p05']
            range_val = norm_stats[f'{feature_name}_range']
            
            before = volume[channel_idx:channel_idx+45].clone()
            for i in range(45):
                ch = volume[channel_idx + i]
                ch_denorm = ((ch + 1.0) / 2.0) * range_val + p05
                volume[channel_idx + i] = ch_denorm
                
            after = volume[channel_idx:channel_idx+45]
            print(f"  features_rest: [{before.min().item():.4f}, {before.max().item():.4f}] -> [{after.min().item():.4f}, {after.max().item():.4f}] (p05={p05:.4f}, range={range_val:.4f})")
            channel_idx += 45
    
    return volume


def _fallback_denormalize(volume, include_features):
    """
    Fallback denormalization when no stats are available.
    Uses reasonable assumptions about gaussian parameter ranges.
    """
    print("Using fallback denormalization (no stats available)")
    volume = volume.clone()
    channel_idx = 0
    
    for feature_name in include_features:
        if feature_name == "opacity":
            # Map from [-1, 1] to [0, 1]
            volume[channel_idx] = (volume[channel_idx] + 1.0) / 2.0
            volume[channel_idx] = th.clamp(volume[channel_idx], 0.0, 1.0)
            channel_idx += 1
            
        elif feature_name == "scaling":
            # Map from [-1, 1] to log space for reasonable scaling values
            # Target exp(log_scale) to be in range [0.01, 10.0], so log_scale in [-4.6, 2.3]
            for i in range(3):
                ch = volume[channel_idx + i]
                # Map [-1, 1] to [-4.6, 2.3] 
                log_scale = (ch + 1.0) / 2.0 * 6.9 - 4.6  # Range: [-4.6, 2.3]
                volume[channel_idx + i] = log_scale
            channel_idx += 3
            
        elif feature_name == "rotation":
            # Normalize quaternions
            quat_start = channel_idx
            quat_end = channel_idx + 4
            quat = volume[quat_start:quat_end]
            quat_flat = quat.view(4, -1)
            quat_norm = th.norm(quat_flat, dim=0, keepdim=True)
            quat_flat = quat_flat / (quat_norm + 1e-6)
            volume[quat_start:quat_end] = quat_flat.view_as(quat)
            channel_idx += 4
            
        elif feature_name == "features_dc":
            # Map from [-1, 1] to reasonable color range [-1, 1] (keep as is)
            channel_idx += 3
            
        elif feature_name == "features_rest":
            # Map from [-1, 1] to reasonable SH range [-1, 1] (keep as is)  
            channel_idx += 45
    
    return volume 