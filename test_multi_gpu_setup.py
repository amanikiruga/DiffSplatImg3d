#!/usr/bin/env python3
"""
Test script to verify multi-GPU setup for voxel gaussian training.
"""

import os
import sys
import torch

# Add the root directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

def test_mpi_setup():
    """Test MPI setup."""
    try:
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
        world_size = MPI.COMM_WORLD.Get_size()
        print(f"✓ MPI setup successful: rank {rank}/{world_size}")
        return True, rank, world_size
    except ImportError:
        print("✗ MPI not available (mpi4py not installed)")
        return False, 0, 1

def test_cuda_setup():
    """Test CUDA setup."""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print(f"✓ CUDA available: {num_gpus} GPUs")
        print(f"  Current device: {current_device} ({device_name})")
        return True, num_gpus
    else:
        print("✗ CUDA not available")
        return False, 0

def test_dist_util():
    """Test guided_diffusion dist_util."""
    try:
        from guided_diffusion import dist_util
        dist_util.setup_dist()
        device = dist_util.dev()
        print(f"✓ dist_util setup successful, device: {device}")
        return True
    except Exception as e:
        print(f"✗ dist_util setup failed: {e}")
        return False

def test_data_loading():
    """Test distributed data loading."""
    try:
        from guided_diffusion.voxel_gaussian_datasets_clean import load_clean_voxel_gaussian_data
        print("✓ Data loading modules imported successfully")
        return True
    except Exception as e:
        print(f"✗ Data loading import failed: {e}")
        return False

def main():
    print("Testing Multi-GPU Setup for Voxel Gaussian Training")
    print("=" * 60)
    
    # Test MPI
    mpi_available, rank, world_size = test_mpi_setup()
    
    # Test CUDA
    cuda_available, num_gpus = test_cuda_setup()
    
    # Test dist_util
    dist_available = test_dist_util()
    
    # Test data loading
    data_available = test_data_loading()
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  MPI Available: {'✓' if mpi_available else '✗'}")
    print(f"  CUDA Available: {'✓' if cuda_available else '✗'} ({num_gpus} GPUs)")
    print(f"  Distributed Utils: {'✓' if dist_available else '✗'}")
    print(f"  Data Loading: {'✓' if data_available else '✗'}")
    
    if mpi_available and cuda_available and dist_available and data_available:
        print(f"\n🎉 Multi-GPU training ready!")
        if world_size > 1:
            print(f"   Running on rank {rank}/{world_size}")
            print(f"   Recommended command: mpirun -np {num_gpus} python scripts/voxel_gaussian_train_clean.py [args...]")
        else:
            print(f"   For multi-GPU training, use: mpirun -np {num_gpus} python test_multi_gpu_setup.py")
    else:
        print(f"\n❌ Multi-GPU training not ready")
        if not mpi_available:
            print("   Install MPI: pip install mpi4py")
        if not cuda_available:
            print("   CUDA/GPU not available")

if __name__ == "__main__":
    main() 