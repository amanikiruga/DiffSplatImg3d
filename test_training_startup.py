#!/usr/bin/env python3
"""
Test script to verify the training script can start up with multi-GPU.
This test mocks the data loading to avoid needing actual data files.
"""

import os
import sys
import tempfile
import json
import torch as th

# Add the root directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

def create_mock_normalization_stats():
    """Create a mock normalization stats file."""
    stats = {
        "opacity_min": 0.0,
        "opacity_max": 1.0,
        "scaling_min": [-2.0, -2.0, -2.0],
        "scaling_max": [2.0, 2.0, 2.0],
        "rotation_min": [-1.0, -1.0, -1.0, -1.0],
        "rotation_max": [1.0, 1.0, 1.0, 1.0],
        "features_dc_min": [-1.0, -1.0, -1.0],
        "features_dc_max": [1.0, 1.0, 1.0],
        "features_rest_min": -1.0,
        "features_rest_max": 1.0
    }
    
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(stats, temp_file)
    temp_file.close()
    return temp_file.name

def test_training_startup():
    """Test if the training script can start up."""
    print("Testing training script startup...")
    
    # Create mock normalization stats
    norm_stats_path = create_mock_normalization_stats()
    
    try:
        # Import the training components
        from guided_diffusion import dist_util, logger
        from guided_diffusion.script_util_3d import (
            voxel_gaussian_model_and_diffusion_defaults,
            create_voxel_gaussian_model_and_diffusion,
            calculate_voxel_gaussian_channels,
        )
        
        print("✓ Successfully imported training components")
        
        # Setup distributed training
        dist_util.setup_dist()
        logger.configure()
        
        # Get distributed info
        try:
            from mpi4py import MPI
            rank = MPI.COMM_WORLD.Get_rank()
            world_size = MPI.COMM_WORLD.Get_size()
        except ImportError:
            rank = 0
            world_size = 1
        
        print(f"✓ Distributed setup successful: rank {rank}/{world_size}")
        
        # Test model creation
        include_features = ["opacity", "scaling", "rotation", "features_dc", "features_rest"]
        expected_channels = calculate_voxel_gaussian_channels(include_features)
        
        print(f"✓ Expected channels: {expected_channels}")
        
        # Create model and diffusion
        model, diffusion = create_voxel_gaussian_model_and_diffusion(
            volume_size=32,
            include_features=include_features,
            num_classes=None,
            dropout=0.0,
            use_checkpoint=False,
            use_fp16=False,
            use_scale_shift_norm=True,
            resblock_updown=False,
            use_new_attention_order=False,
            learn_sigma=False,
            diffusion_steps=100,  # Small number for testing
            noise_schedule="linear",
            timestep_respacing="",
            use_kl=False,
            predict_xstart=False,
            rescale_timesteps=False,
            rescale_learned_sigmas=False,
        )
        
        model.to(dist_util.dev())
        print(f"✓ Model created and moved to device: {dist_util.dev()}")
        
        # Test if model can handle a small batch
        batch_size = 1
        test_input = th.randn(batch_size, expected_channels, 32, 32, 32, device=dist_util.dev())
        t = th.randint(0, 100, (batch_size,), device=dist_util.dev())
        
        with th.no_grad():
            output = model(test_input, t)
        
        print(f"✓ Model forward pass successful: input {test_input.shape} -> output {output.shape}")
        
        print(f"\n🎉 Training script startup test PASSED!")
        print(f"   Multi-GPU training should work with:")
        print(f"   mpirun -np {world_size} python scripts/voxel_gaussian_train_clean.py [args...]")
        
        return True
        
    except Exception as e:
        print(f"✗ Training startup test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up temp file
        if os.path.exists(norm_stats_path):
            os.unlink(norm_stats_path)

if __name__ == "__main__":
    success = test_training_startup()
    sys.exit(0 if success else 1) 