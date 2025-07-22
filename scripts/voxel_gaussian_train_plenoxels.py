#!/usr/bin/env python3
"""
Plenoxels 3D diffusion training with dense_grid.npz files.

This script trains a 3D diffusion model on 32x32x32 plenoxels data
extracted from the svox2 optimization pipeline, using the exact
normalization and rendering approach from opt.ipynb.

Key features:
- Loads dense_grid.npz files with 28 channels (1 density + 27 SH)
- Uses log1p + z-score normalization for density, z-score for SH
- Supports multi-GPU training
- Extensive logging with rendering of training/test samples
- Unconditional generation from noise
"""

import argparse
import os
import sys
import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import imageio
from datetime import datetime
import wandb
from pathlib import Path
import tqdm
import json
import glob


# Add the root directory to the path so we can import from guided_diffusion
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

LOG_PATH = "./checkpoints-plenoxels-diffusion"
# append with date
LOG_PATH = os.path.join(LOG_PATH, datetime.now().strftime("%Y%m%d_%H%M%S")) 
os.makedirs(LOG_PATH, exist_ok=True)

# Add splatter-image path for rendering
SPLATTER_IMAGE_ROOT = os.environ.get("SPLATTER_IMAGE_ROOT", "/om/user/akiruga/splatter-image")
sys.path.append(SPLATTER_IMAGE_ROOT)
sys.path.append(os.path.join(SPLATTER_IMAGE_ROOT, "experiments/voxel-optimization"))

# Add svox2 path for grid utilities
SVOX2_ROOT = "/om/user/akiruga/svox2"
sys.path.append(SVOX2_ROOT)
sys.path.append(os.path.join(SVOX2_ROOT, "opt"))

from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util_3d import (
    model_and_diffusion_defaults_3d,
    create_model_and_diffusion_3d,
    args_to_dict,
    add_dict_to_argparser,
    str2bool,
)
from guided_diffusion.train_util import TrainLoop
from guided_diffusion import gaussian_diffusion as gd

# Import rendering utilities from splatter-image
try:
    from gaussian_renderer import render_predicted
    from splatter_image_datasets.shapenet_chairs_modified import ShapenetChairsModified
    from utils.general_utils import inverse_sigmoid
    RENDERING_AVAILABLE = True
except ImportError as e:
    raise e
    print(f"Warning: Rendering not available - {e}")
    RENDERING_AVAILABLE = False

# Import svox2 utilities
try:
    import svox2
    SVOX2_AVAILABLE = True
except ImportError as e:
    raise e 
    print(f"Warning: svox2 not available - {e}")
    SVOX2_AVAILABLE = False


from plenoxels_utils import get_reference_grid
from plenoxels_utils import render_video

REFERENCE_GRID, RESAMPLE_CAMERAS = get_reference_grid()

assert REFERENCE_GRID is not None, "Reference grid not found"

# ==================== NORMALIZATION UTILITIES ====================
# Extracted from opt.ipynb cells 24-25

EPS = 1e-6

def log1p_pos(x):
    """Apply log1p to positive values (for density channel)"""
    return th.log1p(x.clamp_min_(0.) + EPS)

def load_norm_stats(stats_path):
    """Load normalization statistics from opt.ipynb preprocessing"""
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Normalization stats not found: {stats_path}")
    
    stats = th.load(stats_path)
    return stats["mu"], stats["std"], stats["amax"]

def normalise_plenoxel(grid_tensor, mu, std, amax):
    """
    Normalize a plenoxel grid using the exact approach from opt.ipynb
    
    Args:
        grid_tensor: [D, H, W, C] tensor with 28 channels
        mu, std, amax: normalization parameters
    
    Returns:
        normalized tensor in [-1, 1] range
    """
    N_CH = 28
    g = grid_tensor.view(-1, N_CH).clone()
    
    # Apply log1p to density channel (index 0)
    g[:, 0] = log1p_pos(g[:, 0])
    
    # Z-score normalization
    g = (g - mu) / std
    
    # Scale by amax to get [-1,1] range
    g = g / amax
    g.clamp_(-1, 1)  # Safety clamp
    
    return g.view_as(grid_tensor)

def denormalise_plenoxel(norm_grid, mu, std, amax):
    """
    Denormalize a normalized plenoxel grid (inverse of normalise_plenoxel)
    
    Args:
        norm_grid: normalized tensor in [-1, 1] range
        mu, std, amax: normalization parameters
    
    Returns:
        denormalized tensor in original scale
    """
    N_CH = 28
    g = norm_grid.view(-1, N_CH).clone()
    
    mu = mu.to(g.device)
    std = std.to(g.device)
    amax = amax.to(g.device)
    
    # Inverse scaling
    g = g * amax
    
    # Inverse z-score
    g = g * std + mu
    
    # Inverse log1p for density channel
    g[:, 0] = th.expm1(g[:, 0])
    
    return g.view_as(norm_grid)


# ==================== RENDERING UTILITIES ====================
# Extracted and adapted from opt.ipynb



def dense_to_sparsegrid(
        dense_grid: th.Tensor,
        device: th.device = None
    ) -> svox2.SparseGrid:
    """
    Reconstruct a fully–dense SparseGrid from a dense tensor.

    Args
    ----
    dense_grid : (X, Y, Z, 1 + basis_dim*3) tensor
        Channel 0 = density, 1: = SH coefficients (flattened RGB·basis_dim).
    template_grid : SparseGrid
        Any existing grid whose radius/center/render‑opts you want to clone.
    device : torch.device, optional
        Target device. Defaults to template_grid’s device.

    Returns
    -------
    new_grid : SparseGrid
        A grid ready to be used with all rendering functions.
    """
    template_grid = REFERENCE_GRID
    if device is None:
        device = dense_grid.device

    # ── sizes ────────────────────────────────────────────────────────────────────
    X, Y, Z, C = dense_grid.shape
    basis_dim = (C - 1) // 3

    # ── flatten & split ─────────────────────────────────────────────────────────
    flat = dense_grid.view(-1, C)               # (N, C)  where N = X*Y*Z
    density_data = flat[:, :1].contiguous()     # (N, 1)
    sh_data      = flat[:, 1:].contiguous()     # (N, basis_dim*3)

    # ── make a brand‑new SparseGrid with identical meta‑data ────────────────────
    new_grid = svox2.SparseGrid(
        reso=(X, Y, Z),
        radius=template_grid.radius.tolist(),
        center=template_grid.center.tolist(),
        basis_type=svox2.BASIS_TYPE_SH,
        basis_dim=basis_dim,
        use_z_order=False,
        device=device,
    )

    # overwrite internal tensors
    new_grid.density_data = nn.Parameter(density_data.to(device))
    new_grid.sh_data      = nn.Parameter(sh_data.to(device))

    # every voxel is active → simple dense mapping
    n_vox = X * Y * Z
    new_grid.links = th.arange(
        n_vox, dtype=th.int32, device=device).view(X, Y, Z)
    new_grid.capacity = n_vox

    # optional: accelerate for CUDA ray‑march kernels
    if new_grid.links.is_cuda:
        new_grid.accelerate()

    # copy render options (step size, background, etc.)
    new_grid.opt = template_grid.opt

    return new_grid

def render_plenoxel_video(dense_grid, cameras, fps=12, crop=1.0, save_video=True, video_prefix="render"):
    """
    Render video from dense plenoxel grid and return frames for logging
    Adapted from opt.ipynb Cell 2 and plenoxels_utils.py render_video
    """
    if not SVOX2_AVAILABLE:
        print("Warning: svox2 not available, returning dummy frames")
        return [np.zeros((128, 128, 3), dtype=np.uint8) for _ in range(len(cameras))]
    
    # Convert to sparse grid
    grid = dense_to_sparsegrid(dense_grid)
    grid.eval()
    
    frames = []
    with th.no_grad():
        for cam in cameras:
            # Optional center-crop so you can render faster mid-training
            w, h = cam.width, cam.height
            if crop < 1.0:
                cam = svox2.Camera(
                    cam.c2w, cam.fx, cam.fy,
                    cam.cx * crop, cam.cy * crop,
                    int(w * crop), int(h * crop),
                    ndc_coeffs=cam.ndc_coeffs
                )
            im = grid.volume_render_image(cam, use_kernel=True)
            im = (im.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            frames.append(im)
    
    # Save video file if requested
    if save_video and len(frames) > 0:
        debug_dir = Path(LOG_PATH) / "debug_images"
        debug_dir.mkdir(exist_ok=True)
        video_path = debug_dir / f"{video_prefix}_video.mp4"
        imageio.mimwrite(str(video_path), frames, fps=fps, macro_block_size=8)
        print(f"✔️  Saved preview video → {video_path}")
    
    grid.train()
    return frames

def create_test_cameras(device="cuda"):
    """Create test cameras for rendering, similar to opt.ipynb setup"""
    if not SVOX2_AVAILABLE:
        return []
    
    # Camera parameters from opt.ipynb
    fov_degrees = 51.98948897809546
    fov_radians = fov_degrees * np.pi / 180.0
    image_size = 128
    focal_length = (image_size / 2.0) / np.tan(fov_radians / 2.0)  # 131.25
    principal_point = image_size / 2.0  # 64.0
    
    # Create circular camera trajectory
    num_views = 60
    cameras = []
    
    for i in range(num_views):
        angle = 2 * np.pi * i / num_views
        radius = 1.0
        
        # Camera position
        x = radius * np.cos(angle)
        y = 0.0
        z = radius * np.sin(angle)
        
        # Look at origin
        eye = th.tensor([x, y, z], dtype=th.float32, device=device)
        target = th.tensor([0.0, 0.0, 0.0], dtype=th.float32, device=device)
        up = th.tensor([0.0, 1.0, 0.0], dtype=th.float32, device=device)
        
        # Create view matrix
        forward = F.normalize(target - eye, dim=0)
        right = F.normalize(th.cross(forward, up), dim=0)
        up = th.cross(right, forward)
        
        c2w = th.eye(4, dtype=th.float32, device=device)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = -forward
        c2w[:3, 3] = eye
        
        cam = svox2.Camera(
            c2w,
            focal_length, focal_length,
            principal_point, principal_point,
            image_size, image_size,
            ndc_coeffs=(-1, -1)  # Default from opt.ipynb
        )
        cameras.append(cam)
    
    return cameras


# ==================== DATASET CLASS ====================

class PlenoxelDataset(th.utils.data.Dataset):
    """
    Dataset for loading dense_grid.npz files from svox2 optimization
    Applies proper normalization as in opt.ipynb
    """
    
    def __init__(self, data_dir, norm_stats_path, grid_size=32, random_flip=False, random_rotate=False):
        self.data_dir = Path(data_dir)
        self.grid_size = grid_size
        self.random_flip = random_flip
        self.random_rotate = random_rotate
        
        # Load normalization stats
        self.mu, self.std, self.amax = load_norm_stats(norm_stats_path)
        
        # Find all dense_grid.npz files
        self.file_paths = sorted(list(self.data_dir.rglob("dense_grid.npz")))
        
        if len(self.file_paths) == 0:
            raise ValueError(f"No dense_grid.npz files found in {data_dir}")
        
        print(f"Found {len(self.file_paths)} plenoxel files in {data_dir}")
        
        # Verify shapes
        sample_data = np.load(self.file_paths[0])
        sample_grid = sample_data["dense_grid"]
        print(f"Sample grid shape: {sample_grid.shape}")
        
        if sample_grid.shape != (grid_size, grid_size, grid_size, 28):
            raise ValueError(f"Expected shape ({grid_size}, {grid_size}, {grid_size}, 28), got {sample_grid.shape}")
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Load dense grid
        file_path = self.file_paths[idx]
        data = np.load(file_path)
        dense_grid = th.from_numpy(data["dense_grid"]).float()  # [D, H, W, C]
        
        # Apply data augmentation
        if self.random_flip and th.rand(1) > 0.5:
            # Random flip along one axis
            axis = th.randint(0, 3, (1,)).item()
            dense_grid = th.flip(dense_grid, dims=[axis])
        
        if self.random_rotate and th.rand(1) > 0.5:
            # Random 90-degree rotation in XY plane
            k = th.randint(1, 4, (1,)).item()
            dense_grid = th.rot90(dense_grid, k=k, dims=[0, 1])
        
        # Normalize using opt.ipynb approach
        normalized_grid = normalise_plenoxel(dense_grid, self.mu, self.std, self.amax)
        
        # Convert to [C, D, H, W] format for 3D convolution
        normalized_grid = normalized_grid.permute(3, 0, 1, 2)
        
        return normalized_grid

def load_plenoxel_data(data_dir, norm_stats_path, batch_size, grid_size=32, 
                      random_flip=False, random_rotate=False, 
                      deterministic=False, num_workers=1):
    """
    Load plenoxel dataset with proper distributed sampling
    """
    dataset = PlenoxelDataset(
        data_dir=data_dir,
        norm_stats_path=norm_stats_path,
        grid_size=grid_size,
        random_flip=random_flip,
        random_rotate=random_rotate
    )
    
    if deterministic:
        loader = th.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True
        )
    else:
        # Use distributed sampler for multi-GPU training
        try:
            from mpi4py import MPI
            rank = MPI.COMM_WORLD.Get_rank()
            world_size = MPI.COMM_WORLD.Get_size()
            
            sampler = th.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=True
            )
            loader = th.utils.data.DataLoader(
                dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, drop_last=True
            )
        except ImportError:
            # Fallback for single GPU
            loader = th.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
            )
    
    # Create infinite iterator
    while True:
        for batch in loader:
            yield batch, {}  # No conditioning for unconditional training


# ==================== TRAINING LOOP ====================

class PlenoxelWandbTrainLoop(TrainLoop):
    """
    Training loop with extensive wandb logging and rendering
    """
    
    def __init__(self, *args, norm_stats_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Load normalization stats
        if norm_stats_path and os.path.exists(norm_stats_path):
            self.mu, self.std, self.amax = load_norm_stats(norm_stats_path)
            print(f"Loaded normalization stats from: {norm_stats_path}")
        else:
            raise ValueError(f"Normalization stats required: {norm_stats_path}")
        
        # Setup distributed info
        try:
            from mpi4py import MPI
            self.rank = MPI.COMM_WORLD.Get_rank()
            self.world_size = MPI.COMM_WORLD.Get_size()
        except ImportError:
            self.rank = 0
            self.world_size = 1
        
        self.is_primary_rank = (self.rank == 0)
        
        # Create test cameras for rendering
        # self.test_cameras = create_test_cameras(device=dist_util.dev())
        # print(f"Created {len(self.test_cameras)} test cameras")
        
        # Custom loss weighting with alpha_bar^2
        self.alpha_bar_tensor = th.tensor(self.diffusion.alphas_cumprod, device=dist_util.dev())
        
    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        
        print(f"Training step: {self.step}, batch shape: {batch.shape}")
        
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].detach().to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].detach().to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            
            def compute_losses():
                noise = th.randn_like(micro)
                x_t = self.diffusion.q_sample(micro, t, noise=noise)
                
                # Log visualizations at log intervals
                if self.step % self.log_interval == 0 and self.is_primary_rank:
                    print("Logging training visualizations...")
                    with th.no_grad():
                        self.log_plenoxel_viz(micro[0], "train/gt_data")
                        self.log_plenoxel_viz(x_t[0], "train/noisy_input", is_noise=True)
                
                terms = {}
                model_output = self.ddp_model(x_t, t, **micro_cond)
                
                # Log denoised output
                if self.step % self.log_interval == 0 and self.is_primary_rank:
                    with th.no_grad():
                        self.log_plenoxel_viz(model_output[0], "train/model_output")
                
                target = {
                    gd.ModelMeanType.PREVIOUS_X: self.diffusion.q_posterior_mean_variance(
                        x_start=micro, x_t=x_t, t=t
                    )[0],
                    gd.ModelMeanType.START_X: micro,
                    gd.ModelMeanType.EPSILON: noise,
                }[self.diffusion.model_mean_type]
                
                # Compute MSE loss
                mse_loss = ((target - model_output) ** 2).mean(dim=list(range(1, len(target.shape))))
                
                # Apply custom weighting: ωt = ᾱ²t
                alpha_bar_t = self.alpha_bar_tensor[t]
                custom_weights = alpha_bar_t ** 2
                
                terms["mse"] = mse_loss
                terms["loss"] = mse_loss * custom_weights
                return terms
            
            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()
            
            loss = (losses["loss"] * weights).mean()
            
            # Log to wandb
            if self.is_primary_rank:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/mse": (losses["mse"] * weights).mean().item(),
                    "train/step": self.step,
                    "train/samples": self.step * self.batch_size * self.world_size,
                })
            
            # Log to logger
            logger.logkv_mean("train_loss", loss.item())
            logger.logkv_mean("train_mse", (losses["mse"] * weights).mean().item())
            
            self.mp_trainer.backward(loss)
    
    def log_plenoxel_viz(self, volume, prefix, is_noise=False):
        """Log visualization of a plenoxel volume with rendering"""
        if not self.is_primary_rank:
            return
        
        volume = volume.detach().clone()  # [C, D, H, W]
        
        try:
            print(f"=== LOGGING {prefix} ===")
            print(f"Volume shape: {volume.shape}")
            print(f"Volume range: [{volume.min().item():.3f}, {volume.max().item():.3f}]")
            
            # Convert to [D, H, W, C] for denormalization and rendering
            volume_dhwc = volume.permute(1, 2, 3, 0)  # [D, H, W, C]
            
            if not is_noise:
                # Denormalize for rendering
                denorm_volume = denormalise_plenoxel(volume_dhwc, self.mu, self.std, self.amax)
                print(f"Denormalized range: [{denorm_volume.min().item():.3f}, {denorm_volume.max().item():.3f}]")
            else:
                # For noise, use raw volume (already in correct format for rendering)
                denorm_volume = volume_dhwc
            
            # Render video frames using multiple camera views
            num_views = min(12, len(RESAMPLE_CAMERAS))  # Use up to 12 views
            frames = render_plenoxel_video(
                denorm_volume, 
                RESAMPLE_CAMERAS[:num_views], 
                fps=5, 
                crop=1.0, 
                save_video=True,
                video_prefix=f"{prefix.replace('/', '_')}_step_{self.step}"
            )
            
            if frames and len(frames) > 0:
                print(f"Rendered {len(frames)} frames")
                
                # Save debug images locally
                debug_dir = Path(os.path.join(LOG_PATH, "debug_images"))
                debug_dir.mkdir(exist_ok=True)
                
                # Save first few frames
                for i, frame in enumerate(frames[:4]):
                    frame_path = debug_dir / f"{prefix.replace('/', '_')}_frame_{i}_step_{self.step}.png"
                    imageio.imwrite(frame_path, frame)
                
                # Create multiview grid
                grid_size = math.ceil(math.sqrt(len(frames)))
                while len(frames) < grid_size * grid_size:
                    frames.append(np.zeros_like(frames[0]))
                
                rows = []
                for i in range(grid_size):
                    row_frames = frames[i*grid_size:(i+1)*grid_size]
                    rows.append(np.concatenate(row_frames, axis=1))
                multiview_grid = np.concatenate(rows, axis=0)
                
                # Save grid and video
                grid_path = debug_dir / f"{prefix.replace('/', '_')}_grid_step_{self.step}.png"
                imageio.imwrite(grid_path, multiview_grid)
                
                video_path = debug_dir / f"{prefix.replace('/', '_')}_video_step_{self.step}.mp4"
                imageio.mimwrite(video_path, frames, fps=5)
                
                print(f"Saved visualization to {debug_dir}")
                
                # Log to wandb
                wandb.log({
                    f"{prefix}_multiview": wandb.Image(multiview_grid),
                    f"{prefix}_video": wandb.Video(np.array(frames).transpose(0, 3, 1, 2), fps=5, format="mp4")
                })
            else:
                print(f"ERROR: No frames rendered for {prefix}")
                
        except Exception as e:
            print(f"ERROR: Failed to log {prefix}: {e}")
            import traceback
            traceback.print_exc()
    
    def log_test_generation(self):
        """Log full denoising from pure noise"""
        if not self.is_primary_rank:
            return
        
        try:
            print("=== GENERATING FROM PURE NOISE ===")
            
            # Create pure noise
            noise_shape = (1, 28, 32, 32, 32)  # [B, C, D, H, W]
            pure_noise = th.randn(noise_shape, device=dist_util.dev())
            
            print(f"Pure noise shape: {pure_noise.shape}")
            print(f"Pure noise range: [{pure_noise.min().item():.3f}, {pure_noise.max().item():.3f}]")
            
            # Log pure noise
            self.log_plenoxel_viz(pure_noise[0], "test/pure_noise", is_noise=True)
            
            # Full denoising
            print("Starting denoising process...")
            with th.no_grad():
                model_fn = self.ddp_model
                denoised_sample = self.diffusion.p_sample_loop(
                    model_fn,
                    noise_shape,
                    device=dist_util.dev(),
                    clip_denoised=True,
                    progress=True
                )
            
            print(f"Denoised sample shape: {denoised_sample.shape}")
            print(f"Denoised sample range: [{denoised_sample.min().item():.3f}, {denoised_sample.max().item():.3f}]")
            
            # Log denoised result
            self.log_plenoxel_viz(denoised_sample[0], "test/denoised_generation")
            
            wandb.log({"test/generation_step": self.step})
            
            print("=== TEST GENERATION COMPLETED ===")
            
        except Exception as e:
            print(f"ERROR: Failed to generate test samples: {e}")
            import traceback
            traceback.print_exc()


# ==================== MAIN FUNCTION ====================

def main():
    args = create_argparser().parse_args()
    
    if args.seed != -1:
        th.manual_seed(args.seed)
        th.cuda.manual_seed(args.seed)
        th.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
    
    th.autograd.set_detect_anomaly(True)
    
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
    
    logger.log(f"Distributed training: rank {rank}/{world_size}")
    
    # Setup wandb on primary rank
    if rank == 0:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run,
            config=args_to_dict(args, model_and_diffusion_defaults_3d().keys()),
            tags=["plenoxels", "3d-diffusion", "svox2", "unconditional", f"world_size_{world_size}"]
        )
        
        wandb.config.update({
            "data_type": "plenoxels_32x32x32",
            "channels": 28,
            "normalization": "log1p_density_zscore_sh",
            "world_size": world_size,
            "effective_batch_size": args.batch_size * world_size,
        })
    
    logger.log("Creating 3D diffusion model...")
    
    # Create 3D model and diffusion for plenoxels (28 channels)
    model, diffusion = create_model_and_diffusion_3d(
        volume_size=args.volume_size,
        in_channels=28,  # 1 density + 27 SH
        out_channels=28,  # Same as input (no learned sigma by default)
        num_classes=args.num_classes if args.class_cond else None,
        dropout=args.dropout,
        use_checkpoint=args.use_checkpoint,
        use_fp16=args.use_fp16,
        use_scale_shift_norm=args.use_scale_shift_norm,
        resblock_updown=args.resblock_updown,
        use_new_attention_order=args.use_new_attention_order,
        learn_sigma=args.learn_sigma,
        diffusion_steps=args.diffusion_steps,
        noise_schedule=args.noise_schedule,
        timestep_respacing=args.timestep_respacing,
        use_kl=args.use_kl,
        predict_xstart=args.predict_xstart,
        rescale_timesteps=args.rescale_timesteps,
        rescale_learned_sigmas=args.rescale_learned_sigmas,
    )
    
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    
    logger.log("Creating plenoxel data loader...")
    
    data = load_plenoxel_data(
        data_dir=args.data_dir,
        norm_stats_path=args.norm_stats_path,
        batch_size=args.batch_size,
        grid_size=args.volume_size,
        random_flip=args.random_flip,
        random_rotate=args.random_rotate,
        deterministic=False,
        num_workers=args.num_workers
    )
    
    logger.log("Starting plenoxel diffusion training...")
    
    train_loop = PlenoxelWandbTrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        norm_stats_path=args.norm_stats_path,
    )
    
    # Enhanced run loop with test generation
    original_run_loop = train_loop.run_loop
    
    def enhanced_run_loop():
        original_run_step = train_loop.run_step
        
        def enhanced_run_step(batch, cond):
            # Check max steps
            if args.max_steps is not None and train_loop.step >= args.max_steps:
                print(f"Reached max_steps ({args.max_steps}), stopping training")
                return False
            
            result = original_run_step(batch, cond)
            
            # Test generation at intervals
            if (args.log_test_generation and 
                train_loop.step > 0 and 
                train_loop.step % args.test_generation_interval == 0):
                print(f"Generating test samples at step {train_loop.step}")
                train_loop.log_test_generation()
            
            return result
        
        train_loop.run_step = enhanced_run_step
        
        # Run with max steps support
        if args.max_steps is not None:
            while train_loop.step < args.max_steps:
                try:
                    batch, cond = next(train_loop.data)
                    if enhanced_run_step(batch, cond) == False:
                        break
                except StopIteration:
                    print("Data iterator exhausted")
                    break
        else:
            return original_run_loop()
    
    train_loop.run_loop = enhanced_run_loop
    train_loop.run_loop()


def create_argparser():
    defaults = model_and_diffusion_defaults_3d()
    defaults.update(dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=3000000,  # 3M iterations
        batch_size=8,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        class_cond=False,
        # Wandb
        wandb_project="plenoxels-diffusion",
        wandb_run="",
        log_test_generation=True,
        test_generation_interval=500,
        # Data augmentation
        random_flip=False,
        random_rotate=False,
        num_workers=1,
    ))
    
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    
    # Plenoxel-specific arguments
    parser.add_argument(
        "--norm_stats_path",
        type=str,
        required=True,
        help="Path to normalization statistics from opt.ipynb preprocessing"
    )
    
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum training steps"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Random seed (-1 for no seed)"
    )
    
    return parser


if __name__ == "__main__":
    main() 