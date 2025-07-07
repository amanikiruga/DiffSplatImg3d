#!/usr/bin/env python3
"""
Example script for training a 3D diffusion model with the specifications mentioned.

This script demonstrates how to run 3D diffusion training with:
- Batch size of 8
- Adam optimizer with initial learning rate of 10^-4
- Linear beta scheduling from 0.0015 to 0.05 at 1000 timesteps
- Custom loss weighting with ωt = ᾱ²t
- Train for 3.0M iterations with decaying LR from 10^-4 to 10^-6
- Voxel grid resolution of 32
"""

import subprocess
import sys
import os

def run_3d_training():
    """
    Run 3D diffusion model training with specified parameters.
    """
    
    # Define training parameters
    script_path = "scripts/volume_train.py"
    
    # Check if data directory exists
    data_dir = "./data/volumes"  # Change this to your actual data directory
    if not os.path.exists(data_dir):
        print(f"Warning: Data directory {data_dir} does not exist.")
        print("Please create it and add your volumetric data (.npy or .npz files)")
        print("Or modify the data_dir parameter below.")
    
    args = [
        "python", script_path,
        
        # Data and basic settings
        "--data_dir", data_dir,
        "--batch_size", "8",
        "--microbatch", "8",  # Same as batch size for this example
        
        # Model architecture (3D U-Net with specified architecture)
        "--volume_size", "32",  # Voxel grid resolution
        "--in_channels", "3",
        "--out_channels", "3",
        
        # Training parameters
        "--lr", "1e-4",  # Initial learning rate 10^-4
        "--lr_anneal_steps", "3000000",  # 3M iterations for LR decay to 10^-6
        "--weight_decay", "0.0",
        
        # Custom diffusion parameters
        "--beta_start", "0.0015",  # Custom beta schedule start
        "--beta_end", "0.05",      # Custom beta schedule end
        "--diffusion_steps", "1000",
        
        # Logging and checkpointing
        "--log_interval", "100",
        "--save_interval", "10000",  # Save every 10k iterations
        
        # Use mixed precision for efficiency
        "--use_fp16", "false",  # Set to true if you have sufficient GPU memory
        
        # EMA settings
        "--ema_rate", "0.9999",
        
        # Schedule sampler
        "--schedule_sampler", "uniform",
    ]
    
    print("Starting 3D diffusion model training...")
    print("Command:", " ".join(args))
    print("\nTraining configuration:")
    print("- Batch size: 8")
    print("- Learning rate: 1e-4 → 1e-6 over 3M iterations")
    print("- Beta schedule: 0.0015 → 0.05 (linear)")
    print("- Voxel resolution: 32³")
    print("- Custom loss weighting: ωt = ᾱ²t")
    print()
    
    try:
        # Run the training script
        subprocess.run(args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with exit code: {e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: Could not find training script at {script_path}")
        print("Make sure you're running this from the project root directory.")
        sys.exit(1)


def create_sample_data():
    """
    Create sample volumetric data for testing (optional).
    """
    import numpy as np
    
    data_dir = "./data/volumes"
    os.makedirs(data_dir, exist_ok=True)
    
    # Create a few sample 3D volumes
    for i in range(10):
        # Create a simple 3D volume (e.g., a sphere or random data)
        volume = np.random.rand(32, 32, 32, 3).astype(np.float32)  # DHWC format
        
        # Add some structure (optional)
        center = 16
        radius = 8
        x, y, z = np.meshgrid(range(32), range(32), range(32), indexing='ij')
        sphere_mask = (x - center)**2 + (y - center)**2 + (z - center)**2 <= radius**2
        
        # Apply sphere structure to one channel
        volume[sphere_mask, 0] = 1.0
        
        # Save volume
        np.save(f"{data_dir}/sample_volume_{i:03d}.npy", volume)
    
    print(f"Created {10} sample volumes in {data_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="3D Diffusion Training Example")
    parser.add_argument(
        "--create_sample_data", 
        action="store_true",
        help="Create sample volumetric data for testing"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/volumes",
        help="Directory containing volumetric data"
    )
    
    args = parser.parse_args()
    
    if args.create_sample_data:
        create_sample_data()
        print("Sample data created. Now you can run training.")
    else:
        run_3d_training() 