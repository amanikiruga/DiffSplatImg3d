#!/usr/bin/env python3
"""
Generate normalization statistics for plenoxel training.

This script processes all dense_grid.npz files from svox2 optimization
and computes normalization statistics as done in opt.ipynb Cell 24.

Usage:
    python generate_plenoxel_norm_stats.py --data_dir /path/to/dense_grids --output norm_stats.pt
"""

import argparse
import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm


# Constants from opt.ipynb
N_CH = 28
EPS = 1e-6

def log1p_pos(x):
    """Apply log1p to positive values (for density channel)"""
    return torch.log1p(x.clamp_min_(0.) + EPS)

def generate_normalization_stats(data_dir, output_path, sample_size=5000):
    """
    Generate normalization statistics using the exact approach from opt.ipynb Cell 24.
    
    Args:
        data_dir: Directory containing dense_grid.npz files
        output_path: Path to save normalization statistics
        sample_size: Number of voxels to sample per file for percentile calculation
    """
    data_dir = Path(data_dir)
    paths = sorted(list(data_dir.rglob("dense_grid.npz")))
    
    if len(paths) == 0:
        raise ValueError(f"No dense_grid.npz files found in {data_dir}")
    
    print(f"Found {len(paths)} dense_grid.npz files")
    
    # ─── PASS 1 ── compute μ and σ in transformed space ────────────────────────────
    print("Pass 1: Computing mean and std...")
    sum_, sum2, count = torch.zeros(N_CH), torch.zeros(N_CH), 0
    
    for p in tqdm(paths, desc="μ/σ pass"):
        g = torch.from_numpy(np.load(p)['dense_grid']).float().view(-1, N_CH)
        
        # Apply log1p to density channel (index 0)
        g[:, 0] = log1p_pos(g[:, 0])
        
        sum_  += g.sum(0)
        sum2 += (g ** 2).sum(0)
        count += g.shape[0]
    
    mu = sum_ / count
    std = torch.sqrt(sum2 / count - mu ** 2).clamp_min_(1e-6)
    
    # ─── PASS 2 ── find max |z| per channel (99.9th percentile) ──────────────────
    print("Pass 2: Computing amax (99.9th percentile)...")
    perc_target = 0.999
    samples = [[] for _ in range(N_CH)]
    
    for p in tqdm(paths, desc="amax pass"):
        g = torch.from_numpy(np.load(p)['dense_grid']).float().view(-1, N_CH)
        if g.shape[0] > sample_size:
            # Random sampling to keep memory usage manageable
            g = g[torch.randperm(g.shape[0])[:sample_size]]
        
        # Apply log1p to density channel
        g[:, 0] = log1p_pos(g[:, 0])
        
        # Z-score normalization
        z = (g - mu) / std
        
        # Collect absolute values for percentile calculation
        for ch in range(N_CH):
            samples[ch].append(z[:, ch].abs())
    
    # Compute 99.9th percentile for each channel
    amax = torch.tensor([
        torch.quantile(torch.cat(samples[ch]), perc_target).clamp_min(1.0)
        for ch in range(N_CH)
    ])
    
    # Save statistics
    stats = {
        'mu': mu,
        'std': std, 
        'amax': amax,
        'num_files': len(paths),
        'total_voxels': count,
        'percentile': perc_target
    }
    
    torch.save(stats, output_path)
    print(f"✓ Saved normalization stats to: {output_path}")
    
    # Print summary
    print(f"\nNormalization Statistics Summary:")
    print(f"Number of files processed: {len(paths)}")
    print(f"Total voxels: {count:,}")
    print(f"Percentile threshold: {perc_target}")
    
    print(f"\nPer-channel statistics:")
    channel_names = ["density"] + [f"sh{i:02d}" for i in range(1, N_CH)]
    for i, name in enumerate(channel_names):
        print(f"{name:8s} → mu={mu[i]:.4f}  std={std[i]:.4f}  amax={amax[i]:.4f}")
    
    return stats

def validate_stats(stats_path, data_dir):
    """
    Validate that the normalization stats work correctly on a sample file.
    """
    print(f"\nValidating normalization stats...")
    
    # Load stats
    stats = torch.load(stats_path)
    mu, std, amax = stats['mu'], stats['std'], stats['amax']
    
    # Load a sample file
    data_dir = Path(data_dir)
    sample_file = next(data_dir.rglob("dense_grid.npz"))
    sample_grid = torch.from_numpy(np.load(sample_file)['dense_grid']).float()
    
    # clamp min of the density of the sample grid to 0
    sample_grid[..., 0] = sample_grid[..., 0].clamp_min(0)
    
    print(f"Testing with sample file: {sample_file}")
    print(f"Original grid shape: {sample_grid.shape}")
    print(f"Original density range: [{sample_grid[..., 0].min():.3f}, {sample_grid[..., 0].max():.3f}]")
    
    # Apply normalization
    g = sample_grid.view(-1, N_CH).clone()
    g[:, 0] = log1p_pos(g[:, 0])  # Transform density
    g = (g - mu) / std            # Z-score
    g = g / amax                  # Scale to [-1,1]
    g.clamp_(-1, 1)              # Safety clamp
    normalized_grid = g.view_as(sample_grid)
    
    print(f"Normalized grid range: [{normalized_grid.min():.3f}, {normalized_grid.max():.3f}]")
    print(f"Normalized density range: [{normalized_grid[..., 0].min():.3f}, {normalized_grid[..., 0].max():.3f}]")
    
    # Test denormalization
    g_denorm = normalized_grid.view(-1, N_CH).clone()
    g_denorm = g_denorm * amax          # Unscale
    g_denorm = g_denorm * std + mu      # Inverse z-score
    g_denorm[:, 0] = torch.expm1(g_denorm[:, 0])  # Inverse log1p
    denorm_grid = g_denorm.view_as(sample_grid)
    
    print(f"Denormalized grid range: [{denorm_grid.min():.3f}, {denorm_grid.max():.3f}]")
    print(f"Denormalized density range: [{denorm_grid[..., 0].min():.3f}, {denorm_grid[..., 0].max():.3f}]")
    
    # Check reconstruction error
    density_error = (denorm_grid[..., 0] - sample_grid[..., 0]).abs().mean()
    sh_error = (denorm_grid[..., 1:] - sample_grid[..., 1:]).abs().mean()
    
    print(f"Reconstruction error - Density: {density_error:.6f}, SH: {sh_error:.6f}")
    
    if density_error < 1e-3 and sh_error < 1e-3:
        print("✓ Normalization/denormalization working correctly!")
    else:
        print("✗ Warning: High reconstruction error!")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Generate plenoxel normalization statistics")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing dense_grid.npz files")
    parser.add_argument("--output", type=str, default="norm_stats.pt",
                        help="Output path for normalization statistics")
    parser.add_argument("--sample_size", type=int, default=5000,
                        help="Number of voxels to sample per file for percentile calculation")
    parser.add_argument("--validate", action="store_true",
                        help="Validate the generated stats on a sample file")
    
    args = parser.parse_args()
    
    # Generate stats
    stats = generate_normalization_stats(args.data_dir, args.output, args.sample_size)
    
    # Validate if requested
    if args.validate:
        validate_stats(args.output, args.data_dir)

if __name__ == "__main__":
    main() 