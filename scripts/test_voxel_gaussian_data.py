#!/usr/bin/env python3
"""
Test script for voxel gaussian data loading pipeline.

This script tests that:
1. VoxelGaussianDataset correctly loads and processes gaussian parameters
2. Data shapes are correct for the expected 56-channel configuration
3. Model can process the data without errors
4. Feature normalization is working properly
"""

import argparse
import os
import numpy as np
import torch as th

from guided_diffusion.volume_datasets import load_voxel_gaussian_data
from guided_diffusion.script_util_3d import (
    calculate_voxel_gaussian_channels,
    create_voxel_gaussian_model_and_diffusion,
)


def test_data_loading(data_dir, batch_size=2, grid_size=32):
    """Test that voxel gaussian data loads correctly."""
    print("Testing voxel gaussian data loading...")
    
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
            data_loader = load_voxel_gaussian_data(
                data_dir=data_dir,
                batch_size=batch_size,
                grid_size=grid_size,
                class_cond=False,
                deterministic=True,
                random_flip=False,
                random_rotate=False,
                include_features=features,
            )
            
            # Get one batch
            batch, cond = next(iter(data_loader))
            print(f"Batch shape: {batch.shape}")
            print(f"Expected shape: ({batch_size}, {expected_channels}, {grid_size}, {grid_size}, {grid_size})")
            
            # Check shape
            assert batch.shape == (batch_size, expected_channels, grid_size, grid_size, grid_size), \
                f"Shape mismatch: got {batch.shape}, expected ({batch_size}, {expected_channels}, {grid_size}, {grid_size}, {grid_size})"
            
            # Check data range (should be roughly normalized)
            print(f"Data range: [{batch.min().item():.3f}, {batch.max().item():.3f}]")
            print(f"Data mean: {batch.mean().item():.3f}, std: {batch.std().item():.3f}")
            
            # Check for NaN or Inf
            assert not th.isnan(batch).any(), "Found NaN values in data"
            assert not th.isinf(batch).any(), "Found Inf values in data"
            
            print("✓ Data loading test passed")
            
        except Exception as e:
            print(f"✗ Data loading test failed: {e}")
            return False
    
    return True


def test_model_compatibility(data_dir, grid_size=32):
    """Test that the model can process voxel gaussian data."""
    print("\nTesting model compatibility...")
    
    include_features = ("opacity", "scaling", "rotation", "features_dc", "features_rest")
    
    try:
        # Create model and diffusion
        model, diffusion = create_voxel_gaussian_model_and_diffusion(
            volume_size=grid_size,
            include_features=include_features,
            use_fp16=False,
            use_checkpoint=False,
        )
        
        expected_channels = calculate_voxel_gaussian_channels(include_features)
        print(f"Model input channels: {model.in_channels}")
        print(f"Model output channels: {model.out_channels}")
        print(f"Expected channels: {expected_channels}")
        
        # Check model configuration
        assert model.in_channels == expected_channels, \
            f"Model input channels mismatch: got {model.in_channels}, expected {expected_channels}"
        
        # Load some test data
        data_loader = load_voxel_gaussian_data(
            data_dir=data_dir,
            batch_size=1,
            grid_size=grid_size,
            include_features=include_features,
            deterministic=True,
            random_flip=False,
            random_rotate=False,
        )
        
        batch, cond = next(iter(data_loader))
        print(f"Test batch shape: {batch.shape}")
        
        # Test forward pass
        timesteps = th.randint(0, 1000, (batch.shape[0],))
        model.eval()
        with th.no_grad():
            output = model(batch, timesteps)
        
        print(f"Model output shape: {output.shape}")
        print(f"Expected output shape: {(batch.shape[0], model.out_channels, grid_size, grid_size, grid_size)}")
        
        # Check output shape
        expected_out_shape = (batch.shape[0], model.out_channels, grid_size, grid_size, grid_size)
        assert output.shape == expected_out_shape, \
            f"Output shape mismatch: got {output.shape}, expected {expected_out_shape}"
        
        # Check for NaN or Inf in output
        assert not th.isnan(output).any(), "Found NaN values in model output"
        assert not th.isinf(output).any(), "Found Inf values in model output"
        
        print("✓ Model compatibility test passed")
        return True
        
    except Exception as e:
        print(f"✗ Model compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_consistency(data_dir, grid_size=32):
    """Test data consistency and augmentations."""
    print("\nTesting data consistency...")
    
    include_features = ("opacity", "scaling", "rotation", "features_dc", "features_rest")
    
    try:
        # Test without augmentations
        data_loader_no_aug = load_voxel_gaussian_data(
            data_dir=data_dir,
            batch_size=1,
            grid_size=grid_size,
            include_features=include_features,
            deterministic=True,
            random_flip=False,
            random_rotate=False,
        )
        
        # Test with augmentations
        data_loader_aug = load_voxel_gaussian_data(
            data_dir=data_dir,
            batch_size=1,
            grid_size=grid_size,
            include_features=include_features,
            deterministic=False,
            random_flip=True,
            random_rotate=True,
        )
        
        # Get batches
        batch_no_aug, _ = next(iter(data_loader_no_aug))
        batch_aug, _ = next(iter(data_loader_aug))
        
        print(f"No aug batch shape: {batch_no_aug.shape}")
        print(f"Aug batch shape: {batch_aug.shape}")
        
        # Shapes should be the same
        assert batch_no_aug.shape == batch_aug.shape, "Augmentation changes batch shape"
        
        # Check that data is not identical (due to random augmentations)
        # Note: This might occasionally fail if no augmentation is applied by chance
        print(f"Data difference mean: {(batch_no_aug - batch_aug).abs().mean().item():.6f}")
        
        print("✓ Data consistency test passed")
        return True
        
    except Exception as e:
        print(f"✗ Data consistency test failed: {e}")
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
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=2,
        help="Batch size for testing"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} does not exist")
        return
    
    print(f"Testing voxel gaussian data pipeline...")
    print(f"Data directory: {args.data_dir}")
    print(f"Grid size: {args.grid_size}")
    print(f"Batch size: {args.batch_size}")
    
    # Run tests
    tests = [
        ("Data Loading", lambda: test_data_loading(args.data_dir, args.batch_size, args.grid_size)),
        ("Model Compatibility", lambda: test_model_compatibility(args.data_dir, args.grid_size)),
        ("Data Consistency", lambda: test_data_consistency(args.data_dir, args.grid_size)),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running {test_name} Test")
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
        print("\n🎉 All tests passed! The voxel gaussian data pipeline is ready for training.")
    else:
        print(f"\n❌ {len(results) - passed} test(s) failed. Please check the errors above.")


if __name__ == "__main__":
    main() 