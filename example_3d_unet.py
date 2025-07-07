#!/usr/bin/env python3
"""
Example script demonstrating how to use the 3D U-Net model.

This script shows how to instantiate the 3D U-Net with the specified architecture:
- 4 scaling blocks with 2 ResNet blocks per scale
- Linear increase from 64 to 256 channels
- Skip attention blocks at scaling factors 2, 4, and 8 with 32 channels per head
"""

import torch
import torch.nn as nn
from guided_diffusion.script_util_3d import (
    create_model_3d,
    create_model_and_diffusion_3d,
    model_and_diffusion_defaults_3d,
)
from tqdm import tqdm


def main():
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Creating 3D U-Net model with specified architecture...")
    
    # Get default parameters
    defaults = model_and_diffusion_defaults_3d()
    print(f"Default parameters: {defaults}")
    
    # Create the 3D U-Net model
    model = create_model_3d(
        volume_size=64,
        in_channels=3,
        out_channels=3,
        num_classes=None,
        dropout=0.0,
        use_checkpoint=False,
        use_fp16=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_new_attention_order=False,
    ).to(device)
    
    print(f"Created 3D U-Net model:")
    print(f"- Architecture: 4 scaling blocks with 2 ResNet blocks per scale")
    print(f"- Channel progression: Linear increase from 64 to 256 channels")
    print(f"- Skip attention: At scaling factors 2, 4, and 8 with 32 channels per head")
    print(f"- Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 1
    channels = 3
    depth, height, width = 32, 32, 32
    timesteps = 10
    
    print(f"\nTesting forward pass...")
    print(f"Input shape: [{batch_size}, {channels}, {depth}, {height}, {width}]")
    
    # Create dummy input
    x = torch.randn(batch_size, channels, depth, height, width).to(device)
    t = torch.randint(0, 1000, (batch_size,)).to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(x, t)
    
    print(f"Output shape: {list(output.shape)}")
    print(f"Forward pass successful!")
    
    # Create model with diffusion process
    print(f"\nCreating model with diffusion process...")
    model_with_diffusion, diffusion = create_model_and_diffusion_3d(
        **defaults
    )
    model_with_diffusion = model_with_diffusion.to(device)
    
    print(f"Diffusion steps: {diffusion.num_timesteps}")
    print(f"Noise schedule: {defaults['noise_schedule']}")
    
    # Test diffusion forward process
    print(f"\nTesting diffusion forward process...")
    for _ in tqdm(range(100)):
        with torch.no_grad():
            # Add noise to the input
            t_diffusion = torch.randint(0, diffusion.num_timesteps, (batch_size,)).to(device)
            noisy_x = diffusion.q_sample(x, t_diffusion)
            
            # Model prediction
            model_output = model_with_diffusion(noisy_x, t_diffusion)
            
            # Calculate loss
            loss = diffusion.training_losses(model_with_diffusion, x, t_diffusion)
            
    print(f"Noisy input shape: {list(noisy_x.shape)}")
    print(f"Model output shape: {list(model_output.shape)}")
    print(f"Training loss: {loss['loss'].item():.6f}")
    print(f"Diffusion process test successful!")


if __name__ == "__main__":
    main() 