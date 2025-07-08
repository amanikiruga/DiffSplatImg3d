#!/usr/bin/env python3
"""
Compute global min-max normalization statistics for voxel gaussian dataset.

This script processes the entire training dataset to compute global min/max values 
for each gaussian feature type, which are then used for consistent normalization
during training and inference.

Usage:
    python scripts/compute_global_norm_stats.py --data_dir /path/to/voxel/data --output_path global_norm_stats.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
import torch as th
import numpy as np
from tqdm import tqdm

# Add the root directory to the path so we can import from guided_diffusion
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

def load_gaussian_data(gaussians_path):
    """Load gaussian parameters from a single object."""
    try:
        gaussians = th.load(gaussians_path, map_location="cpu")
        return gaussians
    except Exception as e:
        print(f"Warning: Failed to load {gaussians_path}: {e}")
        return None

def collect_all_features(data_dir, include_features):
    """
    Collect all gaussian features from the entire dataset.
    
    Returns:
        Dict with feature_name -> list of tensors
    """
    print(f"Scanning dataset directory: {data_dir}")
    
    # Find all object directories (containing gaussians.pt)
    object_dirs = []
    for root, dirs, files in os.walk(data_dir):
        if "gaussians.pt" in files:
            object_dirs.append(root)
    
    print(f"Found {len(object_dirs)} objects with gaussians.pt")
    
    # Collect all features
    all_features = {feature_name: [] for feature_name in include_features}
    
    for obj_dir in tqdm(object_dirs, desc="Loading gaussian data"):
        gaussians_path = os.path.join(obj_dir, "gaussians.pt")
        gaussians = load_gaussian_data(gaussians_path)
        
        if gaussians is None:
            continue
            
        # Extract features according to include_features
        for feature_name in include_features:
            if feature_name == "opacity":
                # [N, 1] -> flatten to [N]
                opacity = gaussians["opacity"].float().flatten()
                all_features[feature_name].append(opacity)
                
            elif feature_name == "scaling":
                # [N, 3] -> keep as [N, 3] for per-channel stats
                scaling = gaussians["scaling"].float()
                all_features[feature_name].append(scaling)
                
            elif feature_name == "rotation":
                # [N, 4] -> keep as [N, 4] (we don't normalize rotations)
                rotation = gaussians["rotation"].float()
                all_features[feature_name].append(rotation)
                
            elif feature_name == "features_dc":
                # [N, 1, 3] -> [N, 3]
                features_dc = gaussians["features_dc"].squeeze(1).float()
                all_features[feature_name].append(features_dc)
                
            elif feature_name == "features_rest":
                # [N, 15, 3] -> [N*15*3] (flatten for global stats)
                features_rest = gaussians["features_rest"].float()
                features_rest_flat = features_rest.flatten()
                all_features[feature_name].append(features_rest_flat)
    
    return all_features

def compute_global_min_max_stats(all_features):
    """
    Compute global min-max statistics for normalization.
    
    Based on the user's approach:
    - opacity: min-max to [-1,1] 
    - scaling: log first, then min-max to [-1,1]
    - features_dc: min-max to [-1,1]
    - features_rest: global min-max to [-1,1]
    - rotation: no normalization (quaternions are normalized separately)
    """
    
    stats = {}
    
    for feature_name, feature_tensors in all_features.items():
        if not feature_tensors:
            print(f"Warning: No data found for feature {feature_name}")
            continue
            
        print(f"\nProcessing {feature_name}...")
        
        if feature_name == "opacity":
            # Concatenate all opacity values and compute global min/max
            all_opacity = th.cat(feature_tensors, dim=0)
            print(f"  Total opacity values: {all_opacity.numel()}")
            print(f"  Raw range: [{all_opacity.min().item():.6f}, {all_opacity.max().item():.6f}]")
            
            stats[f"{feature_name}_min"] = all_opacity.min().item()
            stats[f"{feature_name}_max"] = all_opacity.max().item()
            
        elif feature_name == "scaling":
            # Concatenate all scaling values and compute min/max after log transform
            all_scaling = th.cat(feature_tensors, dim=0)  # [N_total, 3]
            print(f"  Total scaling values: {all_scaling.shape}")
            print(f"  Raw range: [{all_scaling.min().item():.6f}, {all_scaling.max().item():.6f}]")
            
            # Apply log transform first (user's approach)
            log_scaling = th.log(th.clamp(all_scaling, min=1e-8))
            print(f"  Log scaling range: [{log_scaling.min().item():.6f}, {log_scaling.max().item():.6f}]")
            
            # Global min/max across all channels
            stats[f"{feature_name}_log_min"] = log_scaling.min().item()
            stats[f"{feature_name}_log_max"] = log_scaling.max().item()
            
        elif feature_name == "rotation":
            # We don't normalize rotations (they get normalized to unit quaternions)
            all_rotation = th.cat(feature_tensors, dim=0)  # [N_total, 4]
            print(f"  Total rotation values: {all_rotation.shape}")
            print(f"  Raw range: [{all_rotation.min().item():.6f}, {all_rotation.max().item():.6f}]")
            # No stats needed - rotations are handled by quaternion normalization
            
        elif feature_name == "features_dc":
            # Concatenate all features_dc and compute global min/max
            all_dc = th.cat(feature_tensors, dim=0)  # [N_total, 3]
            print(f"  Total features_dc values: {all_dc.shape}")
            print(f"  Raw range: [{all_dc.min().item():.6f}, {all_dc.max().item():.6f}]")
            
            stats[f"{feature_name}_min"] = all_dc.min().item()
            stats[f"{feature_name}_max"] = all_dc.max().item()
            
        elif feature_name == "features_rest":
            # Concatenate all features_rest (already flattened) and compute global min/max
            all_rest = th.cat(feature_tensors, dim=0)  # [N_total*15*3]
            print(f"  Total features_rest values: {all_rest.numel()}")
            print(f"  Raw range: [{all_rest.min().item():.6f}, {all_rest.max().item():.6f}]")
            
            stats[f"{feature_name}_min"] = all_rest.min().item()
            stats[f"{feature_name}_max"] = all_rest.max().item()
    
    return stats

def save_stats(stats, output_path):
    """Save statistics to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved normalization statistics to: {output_path}")
    
    # Print summary
    print("\n=== COMPUTED GLOBAL NORMALIZATION STATS ===")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print("=== END STATS ===")

def main():
    parser = argparse.ArgumentParser(description="Compute global normalization statistics for voxel gaussian dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--output_path", type=str, default="global_norm_stats.json", help="Output path for statistics JSON")
    parser.add_argument(
        "--include_features", 
        nargs='+',
        default=["opacity", "scaling", "rotation", "features_dc", "features_rest"],
        choices=["opacity", "scaling", "rotation", "features_dc", "features_rest"],
        help="Gaussian features to include"
    )
    
    args = parser.parse_args()
    
    print("Computing global min-max normalization statistics...")
    print(f"Data directory: {args.data_dir}")
    print(f"Output path: {args.output_path}")
    print(f"Features: {args.include_features}")
    
    # Collect all features from dataset
    all_features = collect_all_features(args.data_dir, args.include_features)
    
    # Compute global min-max statistics
    stats = compute_global_min_max_stats(all_features)
    
    # Save to JSON file
    save_stats(stats, args.output_path)
    
    print("\nDone!")

if __name__ == "__main__":
    main() 