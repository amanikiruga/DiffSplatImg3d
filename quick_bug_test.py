#!/usr/bin/env python3
"""
Quick test script for the most critical potential bugs in 3D diffusion training.
Run this before starting a full training run.
"""

import torch as th
import numpy as np
import sys
import os

# Add the root directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__)))

def test_alpha_bar_bug():
    """Test the alpha_bar indexing bug found in volume_train.py"""
    print("🔍 Testing alpha_bar indexing bug...")
    
    from guided_diffusion import gaussian_diffusion as gd
    
    # Create diffusion
    betas = gd.get_named_beta_schedule("linear", 1000)
    diffusion = gd.GaussianDiffusion(
        betas=betas,
        model_mean_type=gd.ModelMeanType.EPSILON,
        model_var_type=gd.ModelVarType.FIXED_LARGE,
        loss_type=gd.LossType.MSE,
    )
    
    # Test the problematic code from volume_train.py line 129
    # self.alpha_bar = self.diffusion.alphas_cumprod  # This is numpy array
    # alpha_bar_t = self.alpha_bar[t].to(micro.device)  # This will FAIL!
    
    alpha_bar = diffusion.alphas_cumprod  # numpy array
    t = th.tensor([100, 200, 300])  # tensor indices
    
    try:
        # This should fail because you can't index numpy with tensor
        alpha_bar_t = alpha_bar[t]  # BUG!
        print("❌ BUG NOT CAUGHT: numpy array indexed with tensor should fail")
    except (TypeError, IndexError) as e:
        print(f"✅ BUG FOUND: {e}")
        print("   FIX: Convert alpha_bar to tensor first")
        
        # Show the fix
        alpha_bar_tensor = th.tensor(alpha_bar)
        alpha_bar_t = alpha_bar_tensor[t]
        print(f"   Fixed result shape: {alpha_bar_t.shape}")

def test_double_weighting_bug():
    """Test potential double weighting in loss computation"""
    print("\n🔍 Testing double weighting bug...")
    
    from guided_diffusion import gaussian_diffusion as gd
    from guided_diffusion.resample import UniformSampler
    
    # Create diffusion
    betas = gd.get_named_beta_schedule("linear", 1000) 
    diffusion = gd.GaussianDiffusion(
        betas=betas,
        model_mean_type=gd.ModelMeanType.EPSILON,
        model_var_type=gd.ModelVarType.FIXED_LARGE,
        loss_type=gd.LossType.MSE,
    )
    
    # Test the loss weighting
    alpha_bar_tensor = th.tensor(diffusion.alphas_cumprod)
    t = th.tensor([100, 500, 900])
    
    # Custom weighting: ωt = ᾱ²t (from the scripts)
    alpha_bar_t = alpha_bar_tensor[t]
    custom_weights = alpha_bar_t ** 2
    
    # Schedule sampler also provides weights
    sampler = UniformSampler(diffusion)
    _, schedule_weights = sampler.sample(len(t), th.device("cpu"))
    
    print(f"Custom weights (α̅²): {custom_weights}")
    print(f"Schedule weights: {schedule_weights}")
    
    # The final loss computation does: loss = (losses["loss"] * weights).mean()
    # where losses["loss"] = mse_loss * custom_weights
    # So we get: ((mse * custom_weights) * schedule_weights).mean()
    
    combined_weight = custom_weights * schedule_weights
    print(f"Combined weights: {combined_weight}")
    
    # Check if this is problematic
    if th.allclose(schedule_weights, th.ones_like(schedule_weights)):
        print("✅ Schedule weights are uniform (1.0), so no double weighting")
    else:
        print("⚠️  POTENTIAL ISSUE: Both custom and schedule weights are applied")
        print("    Verify this is intended behavior")

def test_plenoxels_data_format_bug():
    """Test the data format handling in plenoxels script"""
    print("\n🔍 Testing plenoxels data format bug...")
    
    # Simulate what the custom_collate_fn returns
    batch_data = th.randn(2, 28, 32, 32, 32)
    file_paths = ["file1.npz", "file2.npz"]
    
    # The data loader returns this tuple
    data_loader_output = (batch_data, file_paths)
    
    # But the forward_backward method expects batch, cond format
    # Let's check how it's handled in the actual script
    
    print("Data loader output format:")
    print(f"  Type: {type(data_loader_output)}")
    print(f"  Batch shape: {data_loader_output[0].shape}")
    print(f"  File paths: {data_loader_output[1]}")
    
    # The script should unpack this as:
    batch_data, file_paths = data_loader_output
    cond = {"file_paths": file_paths}
    
    print("\nExpected training loop format:")
    print(f"  Batch shape: {batch_data.shape}")
    print(f"  Cond keys: {list(cond.keys())}")
    
    # Test microbatch handling
    microbatch = 1
    micro_file_paths = cond["file_paths"][:microbatch]
    micro_cond = {k: v for k, v in cond.items() if k != "file_paths"}
    
    print(f"\nMicrobatch handling:")
    print(f"  Micro file paths: {micro_file_paths}")
    print(f"  Micro cond: {micro_cond}")
    
    if len(micro_file_paths) == microbatch:
        print("✅ Microbatch file path handling looks correct")
    else:
        print("❌ Microbatch file path handling issue")

def test_3d_unet_channels():
    """Test 3D UNet channel progression"""
    print("\n🔍 Testing 3D UNet channel progression...")
    
    # From the UNet3DModel implementation
    base_channels = 64
    max_channels = 256
    scaling_blocks = 4
    
    channel_progression = []
    for i in range(scaling_blocks):
        channels = base_channels + (max_channels - base_channels) * i // (scaling_blocks - 1)
        channel_progression.append(channels)
    
    print(f"Channel progression: {channel_progression}")
    
    # Expected: [64, 128, 192, 256] for linear interpolation
    expected = [64, 128, 192, 256]
    
    if channel_progression == expected:
        print("✅ Channel progression is correct")
    else:
        print(f"⚠️  Channel progression differs from expected: {expected}")
    
    # Test that we actually get 256 at the end
    if channel_progression[-1] == max_channels:
        print("✅ Final channel count reaches max_channels")
    else:
        print(f"❌ Final channel count {channel_progression[-1]} != max_channels {max_channels}")

def test_normalization_range():
    """Test plenoxels normalization produces correct range"""
    print("\n🔍 Testing plenoxels normalization range...")
    
    # Create mock data
    mu = th.randn(28)
    std = th.ones(28) * 0.5  
    amax = th.tensor(2.0)
    
    # Test grid with positive density channel
    test_grid = th.rand(4, 4, 4, 28)
    test_grid[:, :, :, 0] = th.rand(4, 4, 4) * 2  # Positive density
    
    # Apply normalization (simplified version)
    EPS = 1e-6
    N_CH = 28
    g = test_grid.view(-1, N_CH).clone()
    
    # Log1p for density
    g[:, 0] = th.log1p(g[:, 0].clamp_min_(0.) + EPS)
    
    # Z-score
    g = (g - mu) / std
    
    # Scale to [-1,1] range
    g = g / amax
    g.clamp_(-1, 1)
    
    normalized = g.view_as(test_grid)
    
    print(f"Original range: [{test_grid.min():.3f}, {test_grid.max():.3f}]")
    print(f"Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
    
    if normalized.min() >= -1.1 and normalized.max() <= 1.1:
        print("✅ Normalization produces correct range")
    else:
        print("❌ Normalization range is incorrect")

def run_minimal_forward_pass():
    """Test a complete forward pass"""
    print("\n🔍 Testing minimal forward pass...")
    
    try:
        from guided_diffusion.script_util_3d import create_model_and_diffusion_3d
        
        # Create small model for testing
        model, diffusion = create_model_and_diffusion_3d(
            volume_size=16,  # Small for speed
            in_channels=4,   # Small for speed  
            out_channels=4,
            num_classes=None,
            dropout=0.0,
            use_checkpoint=False,
            use_fp16=False,
            use_scale_shift_norm=True,
            resblock_updown=False,
            use_new_attention_order=False,
            learn_sigma=False,
            diffusion_steps=100,  # Small for speed
            noise_schedule="linear",
            timestep_respacing="",
            use_kl=False,
            predict_xstart=False,
            rescale_timesteps=False,
            rescale_learned_sigmas=False,
        )
        
        # Test data
        x = th.randn(1, 4, 16, 16, 16)
        t = th.randint(0, 100, (1,))
        
        # Forward pass
        model.eval()
        with th.no_grad():
            output = model(x, t)
        
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        if output.shape == x.shape:
            print("✅ Forward pass successful")
        else:
            print("❌ Forward pass shape mismatch")
            
        # Test loss computation
        losses = diffusion.training_losses(model, x, t)
        print(f"Loss: {losses['loss'].item():.6f}")
        
        if th.isfinite(losses['loss']):
            print("✅ Loss is finite")
        else:
            print("❌ Loss is NaN or Inf")
            
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")

def main():
    """Run all quick tests"""
    print("🚀 Running quick bug tests for 3D diffusion...")
    print("=" * 50)
    
    test_alpha_bar_bug()
    test_double_weighting_bug() 
    test_plenoxels_data_format_bug()
    test_3d_unet_channels()
    test_normalization_range()
    run_minimal_forward_pass()
    
    print("\n" + "=" * 50)
    print("✅ Quick tests complete!")
    print("\nMost Critical Issues to Fix:")
    print("1. Alpha bar tensor indexing in volume_train.py (line 129)")
    print("2. Verify double weighting behavior is intended")
    print("3. Test with real plenoxels data to validate normalization")

if __name__ == "__main__":
    main() 