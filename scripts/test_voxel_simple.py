#!/usr/bin/env python3
"""
Simple test script for voxel gaussian dataset without MPI dependencies.
"""

import argparse
import os
import sys
import torch as th

# Add the project root to path
sys.path.insert(0, '/om/user/akiruga/diffsplatimg3d')

from guided_diffusion.voxel_gaussian_datasets import VoxelGaussianDataset, _list_voxel_gaussian_dirs_recursively
from guided_diffusion.script_util_3d import calculate_voxel_gaussian_channels


def test_simple_dataset_loading(data_dir, grid_size=32, batch_size=2):
    """Test basic dataset functionality without MPI."""
    print("Testing simple voxel gaussian dataset loading...")
    
    # List available object directories
    object_dirs = _list_voxel_gaussian_dirs_recursively(data_dir)
    print(f"Found {len(object_dirs)} object directories")
    
    if len(object_dirs) == 0:
        print("Error: No voxel gaussian object directories found")
        return False
    
    # Show first few directories
    print("First few directories:")
    for i, obj_dir in enumerate(object_dirs[:5]):
        print(f"  {i+1}: {obj_dir}")
    
    # Test different feature combinations
    feature_combinations = [
        ("opacity",),
        ("opacity", "scaling"),
        ("opacity", "scaling", "rotation"),
        ("opacity", "scaling", "rotation", "features_dc"),
        ("opacity", "scaling", "rotation", "features_dc", "features_rest"),
    ]
    
    for features in feature_combinations:
        print(f"\nTesting features: {features}")
        expected_channels = calculate_voxel_gaussian_channels(features)
        print(f"Expected channels: {expected_channels}")
        
        try:
            # Create dataset directly (without MPI)
            dataset = VoxelGaussianDataset(
                grid_size=grid_size,
                object_dirs=object_dirs[:10],  # Test with first 10 objects
                classes=None,
                shard=0,
                num_shards=1,
                random_flip=False,
                random_rotate=False,
                include_features=features,
            )
            
            print(f"Dataset length: {len(dataset)}")
            print(f"Feature channels: {dataset.feature_channels}")
            
            # Test loading one sample
            sample, cond = dataset[0]
            print(f"Sample shape: {sample.shape}")
            print(f"Expected shape: ({expected_channels}, {grid_size}, {grid_size}, {grid_size})")
            
            # Check shape
            expected_shape = (expected_channels, grid_size, grid_size, grid_size)
            assert sample.shape == expected_shape, \
                f"Shape mismatch: got {sample.shape}, expected {expected_shape}"
            
            # Check data range
            print(f"Data range: [{sample.min().item():.3f}, {sample.max().item():.3f}]")
            print(f"Data mean: {sample.mean().item():.3f}, std: {sample.std().item():.3f}")
            
            # Check for NaN or Inf
            assert not th.isnan(sample).any(), "Found NaN values in data"
            assert not th.isinf(sample).any(), "Found Inf values in data"
            
            print("✓ Simple dataset test passed")
            
        except Exception as e:
            print(f"✗ Simple dataset test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True


def test_data_loading_multiple_samples(data_dir, grid_size=32):
    """Test loading multiple samples."""
    print("\nTesting multiple sample loading...")
    
    object_dirs = _list_voxel_gaussian_dirs_recursively(data_dir)
    if len(object_dirs) < 3:
        print("Warning: Need at least 3 object directories for this test")
        return True
    
    features = ("opacity", "scaling", "rotation", "features_dc", "features_rest")
    
    try:
        dataset = VoxelGaussianDataset(
            grid_size=grid_size,
            object_dirs=object_dirs[:5],  # Test with first 5 objects
            classes=None,
            shard=0,
            num_shards=1,
            random_flip=False,
            random_rotate=False,
            include_features=features,
        )
        
        # Test loading multiple samples
        samples = []
        for i in range(min(3, len(dataset))):
            sample, cond = dataset[i]
            samples.append(sample)
            print(f"Sample {i} shape: {sample.shape}")
        
        # Check that samples are different (they should be from different objects)
        if len(samples) >= 2:
            diff = (samples[0] - samples[1]).abs().mean()
            print(f"Difference between samples 0 and 1: {diff.item():.6f}")
            assert diff > 1e-6, "Samples appear to be identical"
        
        print("✓ Multiple sample test passed")
        return True
        
    except Exception as e:
        print(f"✗ Multiple sample test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_augmentations(data_dir, grid_size=32):
    """Test data augmentations."""
    print("\nTesting data augmentations...")
    
    object_dirs = _list_voxel_gaussian_dirs_recursively(data_dir)
    if len(object_dirs) == 0:
        print("Error: No object directories found")
        return False
    
    features = ("opacity", "scaling", "rotation")  # Use fewer features for faster testing
    
    try:
        # Dataset without augmentations
        dataset_no_aug = VoxelGaussianDataset(
            grid_size=grid_size,
            object_dirs=object_dirs[:1],  # Single object
            classes=None,
            shard=0,
            num_shards=1,
            random_flip=False,
            random_rotate=False,
            include_features=features,
        )
        
        # Dataset with augmentations
        dataset_aug = VoxelGaussianDataset(
            grid_size=grid_size,
            object_dirs=object_dirs[:1],  # Same object
            classes=None,
            shard=0,
            num_shards=1,
            random_flip=True,
            random_rotate=True,
            include_features=features,
        )
        
        # Get samples
        sample_no_aug, _ = dataset_no_aug[0]
        sample_aug, _ = dataset_aug[0]
        
        print(f"No aug shape: {sample_no_aug.shape}")
        print(f"Aug shape: {sample_aug.shape}")
        
        # Shapes should be the same
        assert sample_no_aug.shape == sample_aug.shape, "Augmentation changes shape"
        
        print("✓ Augmentation test passed")
        return True
        
    except Exception as e:
        print(f"✗ Augmentation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", 
        required=True,
        help="Path to voxel gaussian dataset directory"
    )
    parser.add_argument(
        "--grid_size", 
        type=int, 
        default=32,
        help="Voxel grid size"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} does not exist")
        return
    
    print(f"Testing voxel gaussian dataset (simple version)...")
    print(f"Data directory: {args.data_dir}")
    print(f"Grid size: {args.grid_size}")
    
    # Run tests
    tests = [
        ("Simple Dataset Loading", lambda: test_simple_dataset_loading(args.data_dir, args.grid_size)),
        ("Multiple Sample Loading", lambda: test_data_loading_multiple_samples(args.data_dir, args.grid_size)),
        ("Augmentation Testing", lambda: test_augmentations(args.data_dir, args.grid_size)),
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
        print("\n🎉 All tests passed! The voxel gaussian dataset is working correctly.")
    else:
        print(f"\n❌ {len(results) - passed} test(s) failed. Please check the errors above.")


if __name__ == "__main__":
    main() 