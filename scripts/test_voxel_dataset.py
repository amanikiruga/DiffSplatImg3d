#!/usr/bin/env python3

import os
import sys
import torch as th
import numpy as np

# Add the root directory to the path so we can import from guided_diffusion
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from guided_diffusion.voxel_gaussian_datasets import VoxelGaussianDataset


def test_dataset_basic(data_dir):
    """Test basic dataset functionality."""
    print("Testing VoxelGaussianDataset basic functionality...")
    
    try:
        # Get list of object directories
        from guided_diffusion.voxel_gaussian_datasets import _list_voxel_gaussian_dirs_recursively
        object_dirs = _list_voxel_gaussian_dirs_recursively(data_dir)
        print(f"Found {len(object_dirs)} object directories")
        
        # Create dataset with minimal features for testing
        dataset = VoxelGaussianDataset(
            grid_size=32,
            object_dirs=object_dirs,
            include_features=['opacity', 'scaling', 'rotation', 'features_dc'],
            shard=0,
            num_shards=1,
            random_flip=False,
            random_rotate=False
        )
        
        print(f"Dataset created successfully with {len(dataset)} samples")
        print(f"Expected channels: {dataset.feature_channels}")
        
        # Test getting first item
        sample, out_dict = dataset[0]
        print(f"Sample shape: {sample.shape}")
        print(f"Sample dtype: {sample.dtype}")
        print(f"Sample range: [{sample.min():.3f}, {sample.max():.3f}]")
        
        # Check for NaN/Inf
        if th.isnan(sample).any():
            print("Warning: Found NaN values in sample")
        if th.isinf(sample).any():
            print("Warning: Found Inf values in sample")
        
        return True
        
    except Exception as e:
        print(f"Error testing dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_features(data_dir):
    """Test different feature combinations."""
    print("\nTesting different feature combinations...")
    
    # Get list of object directories
    from guided_diffusion.voxel_gaussian_datasets import _list_voxel_gaussian_dirs_recursively
    object_dirs = _list_voxel_gaussian_dirs_recursively(data_dir)
    
    feature_sets = [
        ['opacity'],
        ['opacity', 'scaling'],
        ['opacity', 'scaling', 'rotation'],
        ['opacity', 'scaling', 'rotation', 'features_dc'],
        ['opacity', 'scaling', 'rotation', 'features_dc', 'features_rest']
    ]
    
    expected_channels = [1, 4, 8, 11, 56]
    
    for features, expected in zip(feature_sets, expected_channels):
        try:
            dataset = VoxelGaussianDataset(
                grid_size=32,
                object_dirs=object_dirs,
                include_features=features,
                shard=0,
                num_shards=1,
                random_flip=False,
                random_rotate=False
            )
            
            if dataset.feature_channels != expected:
                print(f"❌ Features {features}: got {dataset.feature_channels} channels, expected {expected}")
                return False
            
            sample, _ = dataset[0]
            if sample.shape[0] != expected:
                print(f"❌ Features {features}: sample has {sample.shape[0]} channels, expected {expected}")
                return False
                
            print(f"✅ Features {features}: {expected} channels, shape {sample.shape}")
            
        except Exception as e:
            print(f"❌ Features {features}: {e}")
            return False
    
    return True


def test_dataset_augmentation(data_dir):
    """Test augmentation functionality."""
    print("\nTesting augmentation...")
    
    # Get list of object directories
    from guided_diffusion.voxel_gaussian_datasets import _list_voxel_gaussian_dirs_recursively
    object_dirs = _list_voxel_gaussian_dirs_recursively(data_dir)
    
    # Test without augmentation
    dataset_no_aug = VoxelGaussianDataset(
        grid_size=32,
        object_dirs=object_dirs,
        include_features=['opacity', 'scaling'],
        shard=0,
        num_shards=1,
        random_flip=False,
        random_rotate=False
    )
    
    # Test with augmentation
    dataset_with_aug = VoxelGaussianDataset(
        grid_size=32,
        object_dirs=object_dirs,
        include_features=['opacity', 'scaling'],
        shard=0,
        num_shards=1,
        random_flip=True,
        random_rotate=True
    )
    
    # Get multiple samples to see variation
    sample1, _ = dataset_with_aug[0]
    sample2, _ = dataset_with_aug[0]  # Same index, should get different augmentation
    
    print(f"Sample 1 range: [{sample1.min():.3f}, {sample1.max():.3f}]")
    print(f"Sample 2 range: [{sample2.min():.3f}, {sample2.max():.3f}]")
    
    # They might be the same due to random seed, but shapes should match
    if sample1.shape != sample2.shape:
        print(f"❌ Augmentation: shape mismatch {sample1.shape} vs {sample2.shape}")
        return False
    
    print("✅ Augmentation test passed")
    return True


def test_dataset_batching(data_dir):
    """Test batch loading."""
    print("\nTesting batch loading...")
    
    try:
        # Get list of object directories
        from guided_diffusion.voxel_gaussian_datasets import _list_voxel_gaussian_dirs_recursively
        object_dirs = _list_voxel_gaussian_dirs_recursively(data_dir)
        
        dataset = VoxelGaussianDataset(
            grid_size=32,
            object_dirs=object_dirs,
            include_features=['opacity', 'scaling', 'rotation'],
            shard=0,
            num_shards=1,
            random_flip=False,
            random_rotate=False
        )
        
        # Test multiple samples
        batch_size = 4
        samples = []
        for i in range(batch_size):
            if i >= len(dataset):
                break
            sample, _ = dataset[i]
            samples.append(sample)
        
        if samples:
            batch = th.stack(samples, dim=0)
            print(f"Batch shape: {batch.shape}")
            print(f"Expected: [{len(samples)}, {dataset.feature_channels}, 32, 32, 32]")
            
            expected_shape = (len(samples), dataset.feature_channels, 32, 32, 32)
            if batch.shape != expected_shape:
                print(f"❌ Batch shape mismatch: got {batch.shape}, expected {expected_shape}")
                return False
            
            print("✅ Batch loading test passed")
            return True
        else:
            print("❌ No samples available for batching test")
            return False
            
    except Exception as e:
        print(f"❌ Batch loading error: {e}")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to voxel gaussian dataset")
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Directory {args.data_dir} does not exist")
        return
    
    print(f"Testing VoxelGaussianDataset...")
    print(f"Data directory: {args.data_dir}")
    
    # Run tests
    tests = [
        ("Basic Functionality", lambda: test_dataset_basic(args.data_dir)),
        ("Feature Combinations", lambda: test_dataset_features(args.data_dir)),
        ("Augmentation", lambda: test_dataset_augmentation(args.data_dir)),
        ("Batch Loading", lambda: test_dataset_batching(args.data_dir)),
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
        print("\n🎉 All tests passed! VoxelGaussianDataset is working correctly.")
    else:
        print(f"\n❌ {len(results) - passed} test(s) failed.")


if __name__ == "__main__":
    main() 