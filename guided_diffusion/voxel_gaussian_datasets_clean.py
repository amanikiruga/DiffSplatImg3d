"""
Clean voxel gaussian dataset implementation using global min-max normalization.

This replaces the complex percentile-based normalization with simple min-max
normalization as shown in the user's notebook example.
"""

import os
import random
import torch as th
from torch.utils.data import DataLoader, Dataset
from .gaussian_norm_utils import (
    gaussian_dict_to_volume, 
    normalize_gaussian_volume,
    load_normalization_stats,
    volume_to_gaussian_dict,
    denormalize_gaussian_volume
)


def _list_voxel_gaussian_dirs_recursively(data_dir):
    """Find all directories containing gaussians.pt files."""
    object_dirs = []
    for root, dirs, files in os.walk(data_dir):
        if "gaussians.pt" in files:
            object_dirs.append(root)
    return object_dirs


class CleanVoxelGaussianDataset(Dataset):
    """
    Clean voxel gaussian dataset with global min-max normalization.
    
    Uses pre-computed global normalization statistics for consistent
    normalization across training and inference.
    """
    
    def __init__(
        self,
        grid_size,
        object_dirs,
        norm_stats_path,
        classes=None,
        shard=0,
        num_shards=1,
        random_flip=True,
        random_rotate=True,
        include_features=("opacity", "scaling", "rotation", "features_dc", "features_rest"),
        is_sanity_check=False,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.local_object_dirs = object_dirs[shard:][::num_shards][:2]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_flip = random_flip
        self.random_rotate = random_rotate
        self.include_features = include_features
        self.is_sanity_check = is_sanity_check
        # Load pre-computed global normalization statistics
        self.norm_stats = load_normalization_stats(norm_stats_path)
        
        print(f"CleanVoxelGaussianDataset: {len(self.local_object_dirs)} objects, "
              f"features: {include_features}")
        print(f"Loaded global normalization stats from: {norm_stats_path}")
        
    def __len__(self):
        return len(self.local_object_dirs)
    
    def __getitem__(self, idx):
        obj_dir = self.local_object_dirs[idx]
        gaussians_path = os.path.join(obj_dir, "gaussians.pt")
        
        # Load gaussian parameters
        gaussians = th.load(gaussians_path, map_location="cpu")
        
        # print("just loaded gaussians stats: ")
        # print(f"gaussians path: {gaussians_path}")
        # for key, value in gaussians.items():
        #     if isinstance(value, th.Tensor):
        #         print(f"  {key}: shape={value.shape}, range=[{value.min().item():.3f}, {value.max().item():.3f}], mean={value.mean().item():.3f}")
        
        voxel_centers = gaussians["xyz"]
        # Convert to volume format (no normalization yet)
        volume = gaussian_dict_to_volume(gaussians, self.include_features, self.grid_size)
        
        volume = normalize_gaussian_volume(volume, self.include_features, self.norm_stats)
        
        if self.is_sanity_check:
            volume_denorm = denormalize_gaussian_volume(volume, self.include_features, self.norm_stats)
            for_sanity_dict = volume_to_gaussian_dict(volume_denorm, self.include_features, self.grid_size, voxel_centers)
        
            print("for sanity, the gaussian dict:")
            for key, value in for_sanity_dict.items():
                if isinstance(value, th.Tensor):
                    print(f"  {key}: shape={value.shape}, range=[{value.min().item():.3f}, {value.max().item():.3f}], mean={value.mean().item():.3f}")
            
            exit("sanity check done")
        return volume, {}
        
        # Apply random augmentations before normalization
        if self.random_flip and random.random() < 0.5:
            # Random flip along one spatial axis
            axis = random.choice([1, 2, 3])  # Don't flip channel axis
            volume = th.flip(volume, dims=[axis])
        
        # if self.random_rotate and random.random() < 0.3:
        #     # Random 90-degree rotation in one plane
        #     axes = random.choice([(1, 2), (1, 3), (2, 3)])
        #     k = random.choice([1, 2, 3])
        #     volume = self._rotate_volume_90(volume, k, axes)
        
        # Apply global min-max normalization
        # volume = normalize_gaussian_volume(volume, self.include_features, self.norm_stats)
        # Ensure tensor doesn't require gradients
        volume = volume.detach()
        
        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = th.tensor(self.local_classes[idx], dtype=th.long)
        
        return volume, out_dict
    
    def _rotate_volume_90(self, volume, k, axes):
        """Rotate volume by k*90 degrees in the specified plane."""
        for _ in range(k):
            volume = th.transpose(volume, axes[0], axes[1])
            volume = th.flip(volume, dims=[axes[1]])
        return volume


def load_clean_voxel_gaussian_data(
    *,
    data_dir,
    norm_stats_path,
    batch_size,
    grid_size,
    class_cond=False,
    deterministic=False,
    random_flip=True,
    random_rotate=True,
    include_features=("opacity", "scaling", "rotation", "features_dc", "features_rest"),
    is_sanity_check=False,
):
    """
    Load voxel gaussian data with clean min-max normalization.
    
    Args:
        data_dir: Directory containing gaussian data
        norm_stats_path: Path to global normalization statistics JSON file
        batch_size: Batch size for DataLoader
        grid_size: Voxel grid size (cubic)
        class_cond: Whether to use class conditioning
        deterministic: Whether to use deterministic data loading
        random_flip: Apply random flipping augmentation
        random_rotate: Apply random rotation augmentation  
        include_features: List of gaussian features to include
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    
    if not os.path.exists(norm_stats_path):
        raise ValueError(f"Normalization stats file not found: {norm_stats_path}")
    
    # Import MPI for distributed data loading
    try:
        from mpi4py import MPI
        current_rank = MPI.COMM_WORLD.Get_rank()
        world_size = MPI.COMM_WORLD.Get_size()
    except ImportError:
        # Fallback for single GPU training
        current_rank = 0
        world_size = 1
    
    # Find all object directories
    all_object_dirs = _list_voxel_gaussian_dirs_recursively(data_dir)
    
    if len(all_object_dirs) == 0:
        raise ValueError(f"No gaussian data found in {data_dir}")
    
    print(f"Found {len(all_object_dirs)} objects in {data_dir}")
    print(f"Distributed training: rank {current_rank}/{world_size}")
    
    # Create classes if class conditioning is enabled
    classes = None
    if class_cond:
        # For now, assign classes based on directory structure
        # This can be customized based on your dataset organization
        classes = [hash(obj_dir) % 1000 for obj_dir in all_object_dirs]
    
    # Create dataset with proper distributed sharding
    dataset = CleanVoxelGaussianDataset(
        grid_size=grid_size,
        object_dirs=all_object_dirs,
        norm_stats_path=norm_stats_path,
        classes=classes,
        shard=current_rank,  # Distribute data across GPUs
        num_shards=world_size,
        random_flip=random_flip,
        random_rotate=random_rotate,
        include_features=include_features,
        is_sanity_check=is_sanity_check,
    )
    
    # Create data loader
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
        )
    
    while True:
        yield from loader 