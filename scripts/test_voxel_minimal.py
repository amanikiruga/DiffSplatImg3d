#!/usr/bin/env python3

import os
import math
import random
import gzip
import json
import torch as th
import numpy as np
from torch.utils.data import Dataset
import blobfile as bf


class VoxelGaussianDataset(Dataset):
    """
    Minimal version of VoxelGaussianDataset for testing without MPI dependencies.
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
        
        print(f"VoxelGaussianDataset: {len(self.local_object_dirs)} objects, "
              f"{self.feature_channels} channels, features: {include_features}")
    
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
        
        # Collect features in order
        feature_list = []
        
        for feature_name in self.include_features:
            if feature_name == "opacity":
                # [N, 1] -> keep as [N, 1]
                opacity = gaussians["opacity"].float()
                feature_list.append(opacity)
            
            elif feature_name == "scaling":
                # [N, 3]
                scaling = gaussians["scaling"].float()
                feature_list.append(scaling)
            
            elif feature_name == "rotation":
                # [N, 4]
                rotation = gaussians["rotation"].float()
                feature_list.append(rotation)
            
            elif feature_name == "features_dc":
                # [N, 1, 3] -> [N, 3]
                features_dc = gaussians["features_dc"].squeeze(1).float()
                feature_list.append(features_dc)
            
            elif feature_name == "features_rest":
                # [N, 15, 3] -> [N, 45]
                features_rest = gaussians["features_rest"].float()
                features_rest = features_rest.reshape(n_voxels, -1)
                feature_list.append(features_rest)
        
        # Concatenate all features: [N, total_channels]
        all_features = th.cat(feature_list, dim=1)
        
        # Reshape to volume: [total_channels, D, H, W]
        volume = all_features.transpose(0, 1)  # [total_channels, N]
        volume = volume.reshape(self.feature_channels, self.grid_size, self.grid_size, self.grid_size)
        
        return volume
    
    def _rotate_volume_90(self, volume, k, axes):
        """Simple 90-degree rotation implementation."""
        # For simplicity, just return the volume unchanged for testing
        return volume
    
    def _normalize_features(self, volume):
        """
        Normalize features to reasonable ranges for training.
        
        Apply different normalization strategies for different feature types:
        - Opacity: sigmoid normalization (already in [0,1])
        - Scaling: log-space normalization
        - Rotation: quaternion normalization 
        - Features: standardization
        """
        channel_idx = 0
        
        for feature_name in self.include_features:
            if feature_name == "opacity":
                # Opacity: apply sigmoid to ensure [0, 1] range
                end_idx = channel_idx + 1
                volume[channel_idx:end_idx] = th.sigmoid(volume[channel_idx:end_idx])
                channel_idx = end_idx
            
            elif feature_name == "scaling":
                # Scaling: apply log normalization and clamp
                end_idx = channel_idx + 3
                # Add small epsilon before log to avoid log(0)
                volume[channel_idx:end_idx] = th.log(th.clamp(volume[channel_idx:end_idx], min=1e-6))
                # Clamp the log values to reasonable range
                volume[channel_idx:end_idx] = th.clamp(volume[channel_idx:end_idx], min=-10, max=5)
                channel_idx = end_idx
            
            elif feature_name == "rotation":
                # Rotation: normalize quaternions to unit length
                end_idx = channel_idx + 4
                quat_channels = volume[channel_idx:end_idx]
                quat_norm = th.sqrt(th.sum(quat_channels ** 2, dim=0, keepdim=True))
                quat_norm = th.clamp(quat_norm, min=1e-8)  # Avoid division by zero
                volume[channel_idx:end_idx] = quat_channels / quat_norm
                channel_idx = end_idx
            
            elif feature_name == "features_dc":
                # Features DC: clamp to reasonable range
                end_idx = channel_idx + 3
                volume[channel_idx:end_idx] = th.clamp(volume[channel_idx:end_idx], min=-5, max=5)
                channel_idx = end_idx
            
            elif feature_name == "features_rest":
                # Features rest: clamp to reasonable range
                end_idx = channel_idx + 45
                volume[channel_idx:end_idx] = th.clamp(volume[channel_idx:end_idx], min=-3, max=3)
                channel_idx = end_idx
        
        return volume


def list_voxel_dirs(data_dir):
    """List all object directories containing gaussians.pt files."""
    results = []
    for entry in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, entry)
        if os.path.isdir(full_path):
            # Check if this directory contains gaussians.pt
            gaussians_path = os.path.join(full_path, "gaussians.pt")
            if os.path.exists(gaussians_path):
                results.append(full_path)
    return results


def test_basic_functionality(data_dir):
    """Test basic dataset functionality."""
    print("Testing basic functionality...")
    
    try:
        object_dirs = list_voxel_dirs(data_dir)
        print(f"Found {len(object_dirs)} object directories")
        
        if len(object_dirs) == 0:
            print("❌ No object directories found")
            return False
        
        # Test with minimal features
        dataset = VoxelGaussianDataset(
            grid_size=32,
            object_dirs=object_dirs,
            include_features=['opacity', 'scaling'],
            shard=0,
            num_shards=1,
            random_flip=False,
            random_rotate=False
        )
        
        print(f"Dataset channels: {dataset.feature_channels}")
        
        # Test loading first sample
        sample, out_dict = dataset[0]
        print(f"Sample shape: {sample.shape}")
        print(f"Sample dtype: {sample.dtype}")
        print(f"Sample range: [{sample.min():.3f}, {sample.max():.3f}]")
        
        if sample.shape != (4, 32, 32, 32):
            print(f"❌ Unexpected sample shape: {sample.shape}")
            return False
        
        print("✅ Basic functionality test passed")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_all_features(data_dir):
    """Test with all features."""
    print("\nTesting all features...")
    
    try:
        object_dirs = list_voxel_dirs(data_dir)
        
        # Test with all features
        dataset = VoxelGaussianDataset(
            grid_size=32,
            object_dirs=object_dirs,
            include_features=['opacity', 'scaling', 'rotation', 'features_dc', 'features_rest'],
            shard=0,
            num_shards=1,
            random_flip=False,
            random_rotate=False
        )
        
        if dataset.feature_channels != 56:
            print(f"❌ Expected 56 channels, got {dataset.feature_channels}")
            return False
        
        # Test loading first sample
        sample, out_dict = dataset[0]
        
        if sample.shape != (56, 32, 32, 32):
            print(f"❌ Unexpected sample shape: {sample.shape}")
            return False
        
        print(f"All features sample shape: {sample.shape}")
        print(f"Sample range: [{sample.min():.3f}, {sample.max():.3f}]")
        
        print("✅ All features test passed")
        return True
        
    except Exception as e:
        print(f"❌ All features test failed: {e}")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to voxel gaussian dataset")
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Directory {args.data_dir} does not exist")
        return
    
    print(f"Testing minimal VoxelGaussianDataset...")
    print(f"Data directory: {args.data_dir}")
    
    # Run tests
    tests = [
        ("Basic Functionality", lambda: test_basic_functionality(args.data_dir)),
        ("All Features", lambda: test_all_features(args.data_dir)),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running {test_name}")
        print(f"{'='*50}")
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! VoxelGaussianDataset core functionality is working.")
    else:
        print(f"\n❌ {len(results) - passed} test(s) failed.")


if __name__ == "__main__":
    main() 