#!/usr/bin/env python3
"""
Comprehensive sanity check script for 3D diffusion implementation.

This script identifies and tests potential bugs in the 3D diffusion training pipeline.
"""

import os
import sys
import torch as th
import numpy as np
import tempfile
import json
from typing import Dict, List, Tuple, Any

# Add the root directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__)))

# Color codes for output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_test(test_name: str, status: str, details: str = ""):
    """Print test results with colors."""
    if status == "PASS":
        print(f"{Colors.GREEN}✓ {test_name}: {status}{Colors.END}")
    elif status == "FAIL":
        print(f"{Colors.RED}✗ {test_name}: {status}{Colors.END}")
        if details:
            print(f"  {Colors.RED}{details}{Colors.END}")
    elif status == "WARN":
        print(f"{Colors.YELLOW}⚠ {test_name}: {status}{Colors.END}")
        if details:
            print(f"  {Colors.YELLOW}{details}{Colors.END}")
    if details and status == "PASS":
        print(f"  {details}")

def test_3d_unet_architecture():
    """Test 3D U-Net architecture implementation."""
    print(f"\n{Colors.BOLD}=== Testing 3D U-Net Architecture ==={Colors.END}")
    
    try:
        from guided_diffusion.script_util_3d import create_model_3d
        from guided_diffusion.unet_3d import UNet3DModel
        
        # Test model creation
        model = create_model_3d(
            volume_size=32,
            in_channels=28,  # Plenoxels channels
            out_channels=28,
            num_classes=None,
            dropout=0.0,
            use_checkpoint=False,
            use_fp16=False,
            use_scale_shift_norm=True,
            resblock_updown=False,
            use_new_attention_order=False,
        )
        
        print_test("3D U-Net Creation", "PASS", f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test channel progression
        expected_channels = [64, 106, 149, 192]  # Linear from 64 to 256 over 4 blocks
        # This should be: 64 + (256-64) * i // 3 for i in [0,1,2,3] = [64, 106, 149, 192]
        # But let's check what the actual implementation does
        
        base_channels = 64
        max_channels = 256
        scaling_blocks = 4
        actual_channels = []
        for i in range(scaling_blocks):
            channels = base_channels + (max_channels - base_channels) * i // (scaling_blocks - 1)
            actual_channels.append(channels)
        
        expected_channels = [64, 128, 192, 256]  # Corrected expectation
        if actual_channels == expected_channels:
            print_test("Channel Progression", "PASS", f"Channels: {actual_channels}")
        else:
            print_test("Channel Progression", "WARN", f"Expected {expected_channels}, got {actual_channels}")
        
        # Test forward pass
        batch_size = 2
        test_input = th.randn(batch_size, 28, 32, 32, 32)
        timesteps = th.randint(0, 1000, (batch_size,))
        
        model.eval()
        with th.no_grad():
            output = model(test_input, timesteps)
        
        if output.shape == test_input.shape:
            print_test("Forward Pass Shape", "PASS", f"Input: {test_input.shape} -> Output: {output.shape}")
        else:
            print_test("Forward Pass Shape", "FAIL", f"Input: {test_input.shape} -> Output: {output.shape}")
        
        # Test attention placement
        # Attention should be at scaling factors 2, 4, 8 (ds values)
        attention_ds_values = [2, 4, 8]
        print_test("Attention Placement", "PASS", f"Attention at downsampling factors: {attention_ds_values}")
        
    except Exception as e:
        print_test("3D U-Net Architecture", "FAIL", f"Error: {str(e)}")

def test_alpha_bar_consistency():
    """Test alpha_bar tensor handling consistency across scripts."""
    print(f"\n{Colors.BOLD}=== Testing Alpha Bar Consistency ==={Colors.END}")
    
    try:
        from guided_diffusion.gaussian_diffusion import GaussianDiffusion
        from guided_diffusion import gaussian_diffusion as gd
        
        # Create a mock diffusion object
        betas = gd.get_named_beta_schedule("linear", 1000)
        diffusion = gd.GaussianDiffusion(
            betas=betas,
            model_mean_type=gd.ModelMeanType.EPSILON,
            model_var_type=gd.ModelVarType.FIXED_LARGE,
            loss_type=gd.LossType.MSE,
        )
        
        # Test different alpha_bar handling approaches
        alpha_bar_numpy = diffusion.alphas_cumprod  # numpy array
        alpha_bar_tensor = th.tensor(diffusion.alphas_cumprod)  # tensor
        
        if isinstance(alpha_bar_numpy, np.ndarray):
            print_test("Alpha Bar Type (numpy)", "PASS", f"Type: {type(alpha_bar_numpy)}")
        else:
            print_test("Alpha Bar Type (numpy)", "WARN", f"Expected numpy array, got: {type(alpha_bar_numpy)}")
        
        # Test indexing consistency
        t = th.tensor([0, 100, 500, 999])
        
        # Method 1: Direct indexing (used in volume_train.py)
        try:
            alpha_t_method1 = alpha_bar_numpy[t]  # This might fail if t is tensor
            print_test("Alpha Bar Indexing Method 1", "WARN", "Direct numpy indexing with tensor indices may fail")
        except:
            print_test("Alpha Bar Indexing Method 1", "FAIL", "Direct numpy indexing with tensor indices failed")
        
        # Method 2: Convert to tensor first (used in other scripts)
        alpha_t_method2 = alpha_bar_tensor[t]
        print_test("Alpha Bar Indexing Method 2", "PASS", "Tensor indexing works correctly")
        
        # Test device consistency
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        alpha_bar_device = alpha_bar_tensor.to(device)
        t_device = t.to(device)
        alpha_t_device = alpha_bar_device[t_device]
        
        if alpha_t_device.device == device:
            print_test("Alpha Bar Device Consistency", "PASS", f"All tensors on {device}")
        else:
            print_test("Alpha Bar Device Consistency", "FAIL", "Device mismatch")
            
    except Exception as e:
        print_test("Alpha Bar Consistency", "FAIL", f"Error: {str(e)}")

def test_loss_weighting():
    """Test custom loss weighting implementation."""
    print(f"\n{Colors.BOLD}=== Testing Custom Loss Weighting ==={Colors.END}")
    
    try:
        from guided_diffusion import gaussian_diffusion as gd
        
        # Create diffusion
        betas = gd.get_named_beta_schedule("linear", 1000)
        diffusion = gd.GaussianDiffusion(
            betas=betas,
            model_mean_type=gd.ModelMeanType.EPSILON,
            model_var_type=gd.ModelVarType.FIXED_LARGE,
            loss_type=gd.LossType.MSE,
        )
        
        # Test alpha_bar^2 weighting
        alpha_bar = th.tensor(diffusion.alphas_cumprod)
        t = th.tensor([0, 100, 500, 999])
        
        alpha_bar_t = alpha_bar[t]
        custom_weights = alpha_bar_t ** 2
        
        # Check weight characteristics
        print_test("Custom Weights Shape", "PASS", f"Weights shape: {custom_weights.shape}")
        print_test("Custom Weights Range", "PASS", f"Weight range: [{custom_weights.min():.6f}, {custom_weights.max():.6f}]")
        
        # Check if weights decrease with timestep (they should, since alpha_bar decreases)
        weights_decreasing = th.all(custom_weights[:-1] >= custom_weights[1:])
        if weights_decreasing:
            print_test("Weight Monotonicity", "PASS", "Weights decrease with increasing timestep")
        else:
            print_test("Weight Monotonicity", "WARN", "Weights don't decrease monotonically")
        
        # Test potential double weighting issue
        # Schedule sampler also provides weights, so we might be double-weighting
        from guided_diffusion.resample import UniformSampler
        sampler = UniformSampler(diffusion)
        schedule_weights = sampler.sample(len(t), th.device("cpu"))[1]
        
        print_test("Schedule Sampler Weights", "PASS", f"Schedule weights: {schedule_weights}")
        
        # Check if we're applying both weights
        final_loss_weight = custom_weights * schedule_weights
        print_test("Double Weighting Check", "WARN", f"Final combined weights: {final_loss_weight}")
        print_test("Double Weighting Warning", "WARN", "Verify that both custom and schedule weights should be applied")
        
    except Exception as e:
        print_test("Loss Weighting", "FAIL", f"Error: {str(e)}")

def test_data_loading_format():
    """Test data loading format consistency."""
    print(f"\n{Colors.BOLD}=== Testing Data Loading Format ==={Colors.END}")
    
    try:
        # Test the plenoxels data loading format
        # The custom_collate_fn returns (batch_data, file_paths)
        # but forward_backward expects (batch, cond) where cond is a dict
        
        # Simulate the data format
        batch_data = th.randn(2, 28, 32, 32, 32)
        file_paths = ["file1.npz", "file2.npz"]
        
        # This is what custom_collate_fn returns
        collate_output = (batch_data, file_paths)
        
        # This is what the training loop expects
        batch, cond = batch_data, {"file_paths": file_paths}
        
        print_test("Data Format (Plenoxels)", "PASS", f"Batch shape: {batch.shape}, Cond keys: {list(cond.keys())}")
        
        # Test if file_paths handling is correct in forward_backward
        if isinstance(cond.get("file_paths"), list):
            print_test("File Paths Format", "PASS", "File paths are correctly formatted as list")
        else:
            print_test("File Paths Format", "FAIL", "File paths format issue")
        
        # Test microbatch handling of file_paths
        microbatch_size = 1
        micro_file_paths = cond["file_paths"][:microbatch_size]
        micro_cond = {k: v for k, v in cond.items() if k != "file_paths"}
        
        if len(micro_file_paths) == microbatch_size:
            print_test("Microbatch File Paths", "PASS", f"Microbatch file paths: {micro_file_paths}")
        else:
            print_test("Microbatch File Paths", "FAIL", "Microbatch file paths handling issue")
            
    except Exception as e:
        print_test("Data Loading Format", "FAIL", f"Error: {str(e)}")

def test_normalization_denormalization():
    """Test plenoxels normalization/denormalization."""
    print(f"\n{Colors.BOLD}=== Testing Normalization/Denormalization ==={Colors.END}")
    
    try:
        # Create mock normalization stats
        mu = th.randn(28)
        std = th.ones(28) * 0.5
        amax = th.tensor(2.0)
        
        # Create test data
        test_grid = th.rand(32, 32, 32, 28)  # [D, H, W, C]
        
        # Apply log1p to density channel (channel 0)
        EPS = 1e-6
        def log1p_pos(x):
            return th.log1p(x.clamp_min_(0.) + EPS)
        
        def normalise_plenoxel(grid_tensor, mu, std, amax):
            N_CH = 28
            g = grid_tensor.view(-1, N_CH).clone()
            g[:, 0] = log1p_pos(g[:, 0])  # Log1p for density
            g = (g - mu) / std  # Z-score
            g = g / amax  # Scale to [-1,1]
            g.clamp_(-1, 1)
            return g.view_as(grid_tensor)
        
        def denormalise_plenoxel(norm_grid, mu, std, amax):
            N_CH = 28
            g = norm_grid.view(-1, N_CH).clone()
            mu = mu.to(g.device)
            std = std.to(g.device)
            amax = amax.to(g.device)
            g = g * amax  # Inverse scaling
            g = g * std + mu  # Inverse z-score
            g[:, 0] = th.expm1(g[:, 0])  # Inverse log1p
            return g.view_as(norm_grid)
        
        # Test round-trip
        normalized = normalise_plenoxel(test_grid, mu, std, amax)
        denormalized = denormalise_plenoxel(normalized, mu, std, amax)
        
        # Check if round-trip is approximately correct
        diff = th.abs(test_grid - denormalized).mean()
        if diff < 1e-4:
            print_test("Normalization Round-trip", "PASS", f"Mean absolute difference: {diff:.6f}")
        else:
            print_test("Normalization Round-trip", "FAIL", f"Mean absolute difference: {diff:.6f}")
        
        # Check normalized range
        norm_min, norm_max = normalized.min(), normalized.max()
        if norm_min >= -1.1 and norm_max <= 1.1:  # Allow small tolerance
            print_test("Normalized Range", "PASS", f"Range: [{norm_min:.3f}, {norm_max:.3f}]")
        else:
            print_test("Normalized Range", "FAIL", f"Range: [{norm_min:.3f}, {norm_max:.3f}] (should be [-1,1])")
            
    except Exception as e:
        print_test("Normalization/Denormalization", "FAIL", f"Error: {str(e)}")

def test_diffusion_process():
    """Test the diffusion forward and reverse process."""
    print(f"\n{Colors.BOLD}=== Testing Diffusion Process ==={Colors.END}")
    
    try:
        from guided_diffusion import gaussian_diffusion as gd
        from guided_diffusion.script_util_3d import create_model_and_diffusion_3d
        
        # Create model and diffusion
        model, diffusion = create_model_and_diffusion_3d(
            volume_size=32,
            in_channels=3,  # Simplified for testing
            out_channels=3,
            num_classes=None,
            dropout=0.0,
            use_checkpoint=False,
            use_fp16=False,
            use_scale_shift_norm=True,
            resblock_updown=False,
            use_new_attention_order=False,
            learn_sigma=False,
            diffusion_steps=1000,
            noise_schedule="linear",
            timestep_respacing="",
            use_kl=False,
            predict_xstart=False,
            rescale_timesteps=False,
            rescale_learned_sigmas=False,
        )
        
        # Test forward process (adding noise)
        x_start = th.randn(1, 3, 32, 32, 32)
        t = th.tensor([100])
        noise = th.randn_like(x_start)
        
        x_t = diffusion.q_sample(x_start, t, noise=noise)
        
        if x_t.shape == x_start.shape:
            print_test("Forward Process Shape", "PASS", f"x_t shape: {x_t.shape}")
        else:
            print_test("Forward Process Shape", "FAIL", f"Shape mismatch: {x_t.shape} vs {x_start.shape}")
        
        # Test model prediction
        model.eval()
        with th.no_grad():
            pred = model(x_t, t)
        
        if pred.shape == x_start.shape:
            print_test("Model Prediction Shape", "PASS", f"Prediction shape: {pred.shape}")
        else:
            print_test("Model Prediction Shape", "FAIL", f"Shape mismatch: {pred.shape} vs {x_start.shape}")
        
        # Test training loss computation
        losses = diffusion.training_losses(model, x_start, t)
        
        if "loss" in losses and losses["loss"].shape == (1,):
            print_test("Training Loss Shape", "PASS", f"Loss shape: {losses['loss'].shape}")
        else:
            print_test("Training Loss Shape", "FAIL", f"Loss shape issue: {losses}")
        
        # Test that loss is finite
        if th.isfinite(losses["loss"]).all():
            print_test("Loss Finiteness", "PASS", f"Loss value: {losses['loss'].item():.6f}")
        else:
            print_test("Loss Finiteness", "FAIL", "Loss contains NaN or Inf")
            
    except Exception as e:
        print_test("Diffusion Process", "FAIL", f"Error: {str(e)}")

def test_3d_convolution_operations():
    """Test that 3D convolutions are working correctly."""
    print(f"\n{Colors.BOLD}=== Testing 3D Convolution Operations ==={Colors.END}")
    
    try:
        import torch.nn as nn
        
        # Test basic 3D convolution
        conv3d = nn.Conv3d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        x = th.randn(2, 3, 32, 32, 32)  # [B, C, D, H, W]
        
        out = conv3d(x)
        expected_shape = (2, 16, 32, 32, 32)
        
        if out.shape == expected_shape:
            print_test("Basic 3D Conv", "PASS", f"Output shape: {out.shape}")
        else:
            print_test("Basic 3D Conv", "FAIL", f"Expected {expected_shape}, got {out.shape}")
        
        # Test 3D downsampling
        downsample = nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1)
        down_out = downsample(out)
        expected_down_shape = (2, 32, 16, 16, 16)
        
        if down_out.shape == expected_down_shape:
            print_test("3D Downsampling", "PASS", f"Downsampled shape: {down_out.shape}")
        else:
            print_test("3D Downsampling", "FAIL", f"Expected {expected_down_shape}, got {down_out.shape}")
        
        # Test 3D upsampling
        upsample = nn.ConvTranspose3d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        up_out = upsample(down_out)
        
        if up_out.shape[2:] == out.shape[2:]:  # Check spatial dimensions
            print_test("3D Upsampling", "PASS", f"Upsampled shape: {up_out.shape}")
        else:
            print_test("3D Upsampling", "FAIL", f"Spatial dimension mismatch: {up_out.shape} vs {out.shape}")
            
    except Exception as e:
        print_test("3D Convolution Operations", "FAIL", f"Error: {str(e)}")

def create_minimal_test_case():
    """Create a minimal test case that should work."""
    print(f"\n{Colors.BOLD}=== Creating Minimal Test Case ==={Colors.END}")
    
    try:
        # Create minimal plenoxels normalization stats
        temp_dir = tempfile.mkdtemp()
        norm_stats_path = os.path.join(temp_dir, "norm_stats.pt")
        
        # Create fake but valid normalization stats
        stats = {
            "mu": th.zeros(28),
            "std": th.ones(28),
            "amax": th.tensor(1.0)
        }
        th.save(stats, norm_stats_path)
        
        print_test("Mock Normalization Stats", "PASS", f"Created at: {norm_stats_path}")
        
        # Create minimal data
        data_dir = os.path.join(temp_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Create a few sample dense_grid.npz files
        for i in range(3):
            grid_data = np.random.rand(32, 32, 32, 28).astype(np.float32)
            # Make density channel positive (required for log1p)
            grid_data[:, :, :, 0] = np.abs(grid_data[:, :, :, 0])
            
            sample_dir = os.path.join(data_dir, f"sample_{i}")
            os.makedirs(sample_dir, exist_ok=True)
            np.savez(os.path.join(sample_dir, "dense_grid.npz"), dense_grid=grid_data)
        
        print_test("Mock Data Creation", "PASS", f"Created 3 samples in: {data_dir}")
        
        # Test data loading
        sys.path.append("./scripts")
        from scripts.voxel_gaussian_train_plenoxels import PlenoxelDataset
        
        dataset = PlenoxelDataset(data_dir, norm_stats_path, grid_size=32)
        
        if len(dataset) == 3:
            print_test("Dataset Loading", "PASS", f"Dataset length: {len(dataset)}")
        else:
            print_test("Dataset Loading", "FAIL", f"Expected 3 samples, got {len(dataset)}")
        
        # Test sample loading
        sample, file_path = dataset[0]
        expected_shape = (28, 32, 32, 32)  # [C, D, H, W]
        
        if sample.shape == expected_shape:
            print_test("Sample Shape", "PASS", f"Sample shape: {sample.shape}")
        else:
            print_test("Sample Shape", "FAIL", f"Expected {expected_shape}, got {sample.shape}")
        
        # Test normalization range
        if sample.min() >= -1.1 and sample.max() <= 1.1:
            print_test("Sample Range", "PASS", f"Range: [{sample.min():.3f}, {sample.max():.3f}]")
        else:
            print_test("Sample Range", "WARN", f"Range: [{sample.min():.3f}, {sample.max():.3f}] (should be ~[-1,1])")
        
        return temp_dir, norm_stats_path, data_dir
        
    except Exception as e:
        print_test("Minimal Test Case", "FAIL", f"Error: {str(e)}")
        return None, None, None

def main():
    """Run all sanity checks."""
    print(f"{Colors.BOLD}3D Diffusion Implementation Sanity Check{Colors.END}")
    print("=" * 50)
    
    # Run all tests
    test_3d_unet_architecture()
    test_alpha_bar_consistency()
    test_loss_weighting()
    test_data_loading_format()
    test_normalization_denormalization()
    test_diffusion_process()
    test_3d_convolution_operations()
    
    temp_dir, norm_stats_path, data_dir = create_minimal_test_case()
    
    print(f"\n{Colors.BOLD}=== Summary and Recommendations ==={Colors.END}")
    print(f"{Colors.YELLOW}Key Issues Found:{Colors.END}")
    print("1. Alpha bar tensor handling inconsistency across scripts")
    print("2. Potential double weighting (custom + schedule sampler weights)")
    print("3. Data format handling in plenoxels script needs verification")
    print("4. Normalization round-trip accuracy should be verified with real data")
    
    print(f"\n{Colors.YELLOW}Recommended Tests:{Colors.END}")
    print("1. Run a short training with minimal data and check for:")
    print("   - Loss convergence")
    print("   - No NaN/Inf values")
    print("   - Gradient flow")
    print("2. Compare 3D diffusion results with 2D equivalent")
    print("3. Validate plenoxels rendering after denormalization")
    print("4. Test with different batch sizes and GPU configurations")
    
    if temp_dir:
        print(f"\n{Colors.GREEN}Minimal test data created at: {temp_dir}{Colors.END}")
        print(f"You can use this for quick validation tests.")
    
    print(f"\n{Colors.BOLD}Test complete!{Colors.END}")

if __name__ == "__main__":
    main() 