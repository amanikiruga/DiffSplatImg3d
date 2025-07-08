#!/usr/bin/env python3
"""
Direct test of voxel gaussian files without dependencies.
"""

import os
import torch as th
import gzip
import json


def list_voxel_dirs(data_dir):
    """List directories containing gaussians.pt files."""
    results = []
    for entry in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, entry)
        if os.path.isdir(full_path):
            gaussians_path = os.path.join(full_path, "gaussians.pt")
            if os.path.exists(gaussians_path):
                results.append(full_path)
            else:
                # Recursively search subdirectories
                results.extend(list_voxel_dirs(full_path))
    return results


def calculate_channels(include_features):
    """Calculate total number of channels."""
    channels = 0
    for feature in include_features:
        if feature == "opacity":
            channels += 1
        elif feature == "scaling":
            channels += 3
        elif feature == "rotation":
            channels += 4
        elif feature == "features_dc":
            channels += 3
        elif feature == "features_rest":
            channels += 45
    return channels


def test_file_loading(data_dir):
    """Test loading raw voxel gaussian files."""
    print("Testing direct voxel gaussian file loading...")
    
    # Find voxel directories
    object_dirs = list_voxel_dirs(data_dir)
    print(f"Found {len(object_dirs)} object directories")
    
    if len(object_dirs) == 0:
        print("Error: No voxel gaussian directories found")
        return False
    
    # Show first few
    print("First few directories:")
    for i, obj_dir in enumerate(object_dirs[:5]):
        print(f"  {i+1}: {obj_dir}")
    
    # Test loading first file
    first_dir = object_dirs[0]
    gaussians_path = os.path.join(first_dir, "gaussians.pt")
    meta_path = os.path.join(first_dir, "meta.json.gz")
    
    print(f"\nTesting file: {gaussians_path}")
    
    try:
        # Load gaussian parameters
        gaussians = th.load(gaussians_path, map_location="cpu")
        print("Successfully loaded gaussians.pt")
        
        # Show keys and shapes
        print("Gaussian parameters:")
        for key, value in gaussians.items():
            print(f"  {key}: {value.shape} ({value.dtype})")
        
        # Load metadata if available
        if os.path.exists(meta_path):
            with gzip.open(meta_path, "rb") as f:
                meta = json.loads(f.read().decode())
            print(f"Metadata: {meta}")
        
        # Check expected keys
        expected_keys = ["opacity", "scaling", "rotation", "features_dc", "features_rest", "xyz"]
        missing_keys = [key for key in expected_keys if key not in gaussians]
        if missing_keys:
            print(f"Warning: Missing keys: {missing_keys}")
        
        # Check grid size
        n_voxels = gaussians["opacity"].shape[0]
        grid_size = round(n_voxels ** (1/3))
        expected_voxels = grid_size ** 3
        print(f"Voxels: {n_voxels}, implied grid size: {grid_size}^3 = {expected_voxels}")
        
        if n_voxels != expected_voxels:
            print(f"Warning: Voxel count doesn't match cubic grid")
        
        # Test feature channel calculation
        features = ("opacity", "scaling", "rotation", "features_dc", "features_rest")
        expected_channels = calculate_channels(features)
        print(f"Expected channels for all features: {expected_channels}")
        
        # Test basic processing
        print("\nTesting basic data processing...")
        
        # Collect features
        feature_list = []
        
        # Opacity: [N, 1] -> keep as [N, 1]
        opacity = gaussians["opacity"].float()
        feature_list.append(opacity)
        print(f"Opacity: {opacity.shape}, range: [{opacity.min():.3f}, {opacity.max():.3f}]")
        
        # Scaling: [N, 3]
        scaling = gaussians["scaling"].float()
        feature_list.append(scaling)
        print(f"Scaling: {scaling.shape}, range: [{scaling.min():.3f}, {scaling.max():.3f}]")
        
        # Rotation: [N, 4]
        rotation = gaussians["rotation"].float()
        feature_list.append(rotation)
        print(f"Rotation: {rotation.shape}, range: [{rotation.min():.3f}, {rotation.max():.3f}]")
        
        # Features DC: [N, 1, 3] -> [N, 3]
        features_dc = gaussians["features_dc"].squeeze(1).float()
        feature_list.append(features_dc)
        print(f"Features DC: {features_dc.shape}, range: [{features_dc.min():.3f}, {features_dc.max():.3f}]")
        
        # Features rest: [N, 15, 3] -> [N, 45]
        features_rest = gaussians["features_rest"].float()
        features_rest = features_rest.reshape(n_voxels, -1)
        feature_list.append(features_rest)
        print(f"Features rest: {features_rest.shape}, range: [{features_rest.min():.3f}, {features_rest.max():.3f}]")
        
        # Concatenate all features: [N, total_channels]
        all_features = th.cat(feature_list, dim=1)
        print(f"All features concatenated: {all_features.shape}")
        
        # Reshape to volume: [total_channels, D, H, W]
        volume = all_features.transpose(0, 1)  # [total_channels, N]
        volume = volume.reshape(expected_channels, grid_size, grid_size, grid_size)
        print(f"Volume shape: {volume.shape}")
        print(f"Expected shape: ({expected_channels}, {grid_size}, {grid_size}, {grid_size})")
        
        # Check for NaN or Inf
        if th.isnan(volume).any():
            print("Warning: Found NaN values in volume")
        if th.isinf(volume).any():
            print("Warning: Found Inf values in volume")
        
        print("✓ Direct file loading test passed")
        return True
        
    except Exception as e:
        print(f"✗ Direct file loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_files(data_dir, num_files=3):
    """Test loading multiple files."""
    print(f"\nTesting multiple file loading ({num_files} files)...")
    
    object_dirs = list_voxel_dirs(data_dir)
    if len(object_dirs) < num_files:
        print(f"Warning: Only {len(object_dirs)} files available, testing all")
        num_files = len(object_dirs)
    
    for i in range(num_files):
        obj_dir = object_dirs[i]
        gaussians_path = os.path.join(obj_dir, "gaussians.pt")
        
        try:
            gaussians = th.load(gaussians_path, map_location="cpu")
            n_voxels = gaussians["opacity"].shape[0]
            grid_size = round(n_voxels ** (1/3))
            
            print(f"File {i+1}: {n_voxels} voxels, grid {grid_size}^3")
            
            # Check consistency
            for key in ["opacity", "scaling", "rotation", "features_dc", "features_rest"]:
                if key not in gaussians:
                    print(f"  Warning: Missing {key}")
                    continue
                shape = gaussians[key].shape
                if shape[0] != n_voxels:
                    print(f"  Warning: {key} has {shape[0]} entries, expected {n_voxels}")
        
        except Exception as e:
            print(f"  Error loading file {i+1}: {e}")
            return False
    
    print("✓ Multiple file test passed")
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to voxel gaussian dataset")
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Directory {args.data_dir} does not exist")
        return
    
    print(f"Testing voxel gaussian files directly...")
    print(f"Data directory: {args.data_dir}")
    
    # Run tests
    tests = [
        ("File Loading", lambda: test_file_loading(args.data_dir)),
        ("Multiple Files", lambda: test_multiple_files(args.data_dir)),
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
        print("\n🎉 All tests passed! The voxel gaussian files are valid and ready for processing.")
    else:
        print(f"\n❌ {len(results) - passed} test(s) failed.")


if __name__ == "__main__":
    main() 