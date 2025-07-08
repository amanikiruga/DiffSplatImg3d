#!/usr/bin/env python3
"""
Clean voxel gaussian diffusion training with global min-max normalization.

This script replaces the complex percentile-based normalization with simple
global min-max normalization based on pre-computed statistics.

Training configuration matches the paper:
- Batch size of 8
- Adam optimizer with initial learning rate of 10^-4  
- Linear beta scheduling from 0.0015 to 0.05 at 1000 timesteps
- Custom loss weighting with ωt = ᾱ²t
- Train for 3.0M iterations with decaying LR from 10^-4 to 10^-6
- Voxel grid resolution of 32^3
"""

import argparse
import os
import sys
import numpy as np
import torch as th
import torch.nn.functional as F
import imageio
import wandb
from pathlib import Path

# Add the root directory to the path so we can import from guided_diffusion
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Add splatter-image path for rendering
SPLATTER_IMAGE_ROOT = os.environ.get("SPLATTER_IMAGE_ROOT", "/om/user/akiruga/splatter-image")
sys.path.append(SPLATTER_IMAGE_ROOT)
sys.path.append(os.path.join(SPLATTER_IMAGE_ROOT, "experiments/voxel-optimization"))

from guided_diffusion import dist_util, logger
from guided_diffusion.voxel_gaussian_datasets_clean import load_clean_voxel_gaussian_data
from guided_diffusion.gaussian_norm_utils import (
    denormalize_gaussian_volume,
    volume_to_gaussian_dict,
    load_normalization_stats
)
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util_3d import (
    voxel_gaussian_model_and_diffusion_defaults,
    create_voxel_gaussian_model_and_diffusion,
    calculate_voxel_gaussian_channels,
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
    print(f"Warning: Rendering not available - {e}")
    RENDERING_AVAILABLE = False


class CleanGaussianVoxelGrid(th.nn.Module):
    """
    Clean voxel grid class for rendering using the new normalization approach.
    """
    def __init__(self, grid_size=32, max_sh_degree=3, device="cuda"):
        super().__init__()
        self.grid_size = grid_size
        self.max_sh_degree = max_sh_degree
        
        # Create fixed voxel centers
        half_extent = 0.5
        coords = th.linspace(-half_extent, half_extent, grid_size, device=device)
        x, y, z = th.meshgrid(coords, coords, coords, indexing="ij")
        voxel_centers = th.stack([x, y, z], dim=-1).reshape(grid_size, grid_size, grid_size, 3)
        self.register_buffer("voxel_centers", voxel_centers)

    def load_from_volume(self, volume, include_features, norm_stats):
        """
        Load parameters from a normalized volume tensor using clean denormalization.
        
        Args:
            volume: [C, D, H, W] normalized tensor in [-1,1] range
            include_features: List of feature names
            norm_stats: Global normalization statistics
        """
        # Denormalize the volume
        denorm_volume = denormalize_gaussian_volume(volume.detach(), include_features, norm_stats)
        
        # Convert to gaussian dict
        gauss_dict = volume_to_gaussian_dict(
            denorm_volume, include_features, self.grid_size, 
            self.voxel_centers, opacity_threshold=0.01
        )
        
        return gauss_dict

    def get_gaussian_parameters_at_voxels(self, volume, include_features, norm_stats, opacity_threshold=0.01):
        """Convert normalized volume to gaussian parameters for rendering."""
        return self.load_from_volume(volume, include_features, norm_stats)


def create_render_config():
    """Create a minimal config for rendering."""
    class Config:
        def __init__(self):
            self.data = type('', (), {})()
            self.data.white_background = True
            self.data.training_resolution = 128
            self.data.fov = 45
            self.data.znear = 0.1
            self.data.zfar = 10.0
    return Config()


def render_voxel_views(volume, include_features, norm_stats, dataset_sample, cfg, device, num_views=8, opacity_threshold=0.01):
    """
    Render multiple views of a normalized voxel volume.
    
    Args:
        volume: [C, D, H, W] normalized volume tensor
        include_features: List of feature names
        norm_stats: Global normalization statistics
        dataset_sample: Sample from dataset with camera information
        cfg: Rendering configuration
        device: torch device
        num_views: Number of views to render
        opacity_threshold: Opacity threshold for rendering
    
    Returns:
        List of rendered images as numpy arrays
    """
    if not RENDERING_AVAILABLE:
        print("Warning: Rendering not available, returning dummy images")
        return [np.zeros((128, 128, 3), dtype=np.uint8) for _ in range(num_views)]
    
    bg_color = [1, 1, 1] if cfg.data.white_background else [0, 0, 0]
    background = th.tensor(bg_color, dtype=th.float32, device=device)
    
    # Create voxel grid and get gaussian parameters
    voxel_grid = CleanGaussianVoxelGrid(device=device)
    gauss = voxel_grid.get_gaussian_parameters_at_voxels(volume, include_features, norm_stats, opacity_threshold)
    
    # Select views to render
    total_views = dataset_sample.get('gt_images', th.zeros(25, 3, 128, 128)).shape[0]
    view_indices = th.linspace(0, total_views-1, num_views).long()
    
    frames = []
    with th.no_grad():
        for v in view_indices:
            try:
                if RENDERING_AVAILABLE and 'world_view_transforms_absolute' in dataset_sample:
                    out = render_predicted(
                        gauss,
                        dataset_sample['world_view_transforms_absolute'][v],
                        dataset_sample['full_proj_transforms_absolute'][v], 
                        dataset_sample['camera_centers_absolute'][v],
                        background, cfg
                    )
                    img = out['render'].cpu().numpy().transpose(1, 2, 0)
                    frames.append((np.clip(img, 0, 1) * 255).astype(np.uint8))
                else:
                    # Fallback: create dummy image
                    frames.append(np.zeros((128, 128, 3), dtype=np.uint8))
            except Exception as e:
                print(f"Warning: Rendering view {v} failed: {e}")
                frames.append(np.zeros((128, 128, 3), dtype=np.uint8))
    
    return frames


def create_multiview_grid(frames, grid_size=4):
    """Create a grid of multiview images."""
    if not frames:
        return np.zeros((512, 512, 3), dtype=np.uint8)
    
    # Pad frames to fill grid
    while len(frames) < grid_size * grid_size:
        frames.append(np.zeros_like(frames[0]))
    
    # Take only what we need
    frames = frames[:grid_size * grid_size]
    
    # Arrange in grid
    rows = []
    for i in range(grid_size):
        row_frames = frames[i*grid_size:(i+1)*grid_size]
        rows.append(np.concatenate(row_frames, axis=1))
    
    return np.concatenate(rows, axis=0)


class CleanVoxelGaussianWandbTrainLoop(TrainLoop):
    """
    Clean training loop with wandb logging and minimal normalization complexity.
    """
    
    def __init__(self, *args, include_features=None, norm_stats_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.include_features = include_features or ["opacity", "scaling", "rotation", "features_dc", "features_rest"]
        self.alpha_bar_tensor = th.tensor(self.diffusion.alphas_cumprod, device=dist_util.dev())
        self.render_config = create_render_config()
        
        # Load global normalization stats
        if norm_stats_path and os.path.exists(norm_stats_path):
            self.norm_stats = load_normalization_stats(norm_stats_path)
            print(f"Loaded global normalization stats from: {norm_stats_path}")
        else:
            print("Warning: No normalization stats provided - rendering may fail")
            self.norm_stats = None
        
        # Try to get dataset sample for camera info
        self.dataset_sample = {}
        try:
            # Get a sample from the data loader for camera information
            sample_batch, sample_cond = next(iter(self.data))
            if len(sample_batch) > 0:
                print("Got dataset sample for camera setup")
        except:
            print("Warning: Could not get dataset sample for rendering")
    
    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            # Custom loss computation with alpha_bar^2 weighting
            def compute_losses():
                noise = th.randn_like(micro)
                x_t = self.diffusion.q_sample(micro, t, noise=noise)
                
                # Log noisy input visualization occasionally
                if self.step % self.log_interval == 0 and self.step % 50 == 0:
                    self.log_voxel_viz(x_t[0], f"train/noisy_voxel_viz_step_{self.step}")
                
                terms = {}
                model_output = self.ddp_model(x_t, t, **micro_cond)
                
                # Log denoised output visualization occasionally  
                if self.step % self.log_interval == 0 and self.step % 50 == 0:
                    self.log_voxel_viz(model_output[0], f"train/denoised_voxel_viz_step_{self.step}")
                
                target = {
                    gd.ModelMeanType.PREVIOUS_X: self.diffusion.q_posterior_mean_variance(
                        x_start=micro, x_t=x_t, t=t
                    )[0],
                    gd.ModelMeanType.START_X: micro,
                    gd.ModelMeanType.EPSILON: noise,
                }[self.diffusion.model_mean_type]
                
                assert model_output.shape == target.shape == micro.shape
                
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
            wandb.log({
                "train/loss": loss.item(),
                "train/mse": (losses["mse"] * weights).mean().item(),
                "train/step": self.step,
                "train/samples": self.step * self.batch_size,
            })
            
            # Log losses to logger as well
            logger.logkv_mean("train_loss", loss.item())
            logger.logkv_mean("train_mse", (losses["mse"] * weights).mean().item())
            
            self.mp_trainer.backward(loss)
    
    def log_voxel_viz(self, volume, prefix):
        """Log visualization of a voxel volume."""
        if self.norm_stats is None:
            print(f"Warning: Cannot render {prefix} - no normalization stats")
            return
            
        try:
            # Render views
            frames = render_voxel_views(
                volume, self.include_features, self.norm_stats, 
                self.dataset_sample, self.render_config, dist_util.dev()
            )
            
            if frames:
                # Create multiview grid
                multiview_grid = create_multiview_grid(frames)
                
                # Log to wandb
                wandb.log({
                    f"{prefix}_multiview": wandb.Image(multiview_grid),
                    f"{prefix}_video": wandb.Video(np.array(frames), fps=5, format="mp4")
                })
                
        except Exception as e:
            print(f"Warning: Failed to log {prefix}: {e}")
    
    def log_test_generation(self):
        """Log test-time generation from pure noise."""
        if self.norm_stats is None:
            print("Warning: Cannot generate test samples - no normalization stats")
            return
            
        try:
            print("Generating test samples from pure noise...")
            
            # Sample from pure noise
            noise_shape = (1, 56, 32, 32, 32)  # [B, C, D, H, W]
            pure_noise = th.randn(noise_shape, device=dist_util.dev())
            
            # Log pure noise visualization
            noise_frames = render_voxel_views(
                pure_noise[0], self.include_features, self.norm_stats,
                self.dataset_sample, self.render_config, dist_util.dev()
            )
            
            if noise_frames:
                noise_grid = create_multiview_grid(noise_frames)
                wandb.log({
                    "test/noise_grid": wandb.Image(noise_grid),
                    "test/noise_video": wandb.Video(np.array(noise_frames), fps=5, format="mp4")
                })
            
            # Full denoising process
            with th.no_grad():
                model_fn = self.ddp_model
                denoised_sample = self.diffusion.p_sample_loop(
                    model_fn,
                    noise_shape,
                    device=dist_util.dev(),
                    clip_denoised=True,
                    progress=True
                )
            
            # Log denoised result
            denoised_frames = render_voxel_views(
                denoised_sample[0], self.include_features, self.norm_stats,
                self.dataset_sample, self.render_config, dist_util.dev()
            )
            
            if denoised_frames:
                denoised_grid = create_multiview_grid(denoised_frames)
                wandb.log({
                    "test/denoised_grid_generation": wandb.Image(denoised_grid),
                    "test/denoised_video": wandb.Video(np.array(denoised_frames), fps=5, format="mp4"),
                    "test/generation_step": self.step
                })
            
            print("Test generation logging completed")
            
        except Exception as e:
            print(f"Warning: Failed to log test generation: {e}")
            import traceback
            traceback.print_exc()


def main():
    args = create_argparser().parse_args()

    # Setup wandb
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run,
        config=args_to_dict(args, voxel_gaussian_model_and_diffusion_defaults().keys()),
        tags=["voxel-gaussian", "3d-diffusion", "clean-normalization"]
    )
    
    # Log all training parameters
    wandb.config.update({
        "gaussian_features": args.include_features,
        "expected_channels": calculate_voxel_gaussian_channels(args.include_features),
        "grid_size": args.volume_size,
        "custom_loss_weighting": "alpha_bar_squared",
        "beta_schedule": f"linear_{args.beta_start}_to_{args.beta_end}",
        "normalization_type": "global_min_max",
        "norm_stats_path": args.norm_stats_path,
    })

    dist_util.setup_dist()
    logger.configure()

    # Calculate expected channels
    expected_channels = calculate_voxel_gaussian_channels(args.include_features)
    logger.log(f"Expected input channels: {expected_channels}")
    logger.log(f"Included features: {args.include_features}")
    logger.log(f"Using normalization stats from: {args.norm_stats_path}")

    logger.log("creating voxel gaussian 3D model and diffusion...")
    
    # Create model and diffusion using voxel-specific function
    model, diffusion = create_voxel_gaussian_model_and_diffusion(
        volume_size=args.volume_size,
        include_features=args.include_features,
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

    logger.log("creating clean voxel gaussian data loader...")
    data = load_clean_voxel_gaussian_data(
        data_dir=args.data_dir,
        norm_stats_path=args.norm_stats_path,
        batch_size=args.batch_size,
        grid_size=args.volume_size,
        class_cond=args.class_cond,
        deterministic=False,
        random_flip=args.random_flip,
        random_rotate=args.random_rotate,
        include_features=args.include_features,
    )

    logger.log("training clean voxel gaussian 3D diffusion model...")
    train_loop = CleanVoxelGaussianWandbTrainLoop(
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
        include_features=args.include_features,
        norm_stats_path=args.norm_stats_path,
    )
    
    # Run training with periodic test generation
    original_run_loop = train_loop.run_loop
    
    def enhanced_run_loop():
        # Log test generation at start
        if args.log_test_generation:
            train_loop.log_test_generation()
        
        # Override run_step to add periodic test generation and max_steps
        original_run_step = train_loop.run_step
        
        def enhanced_run_step(batch, cond):
            # Check if we've reached max_steps
            if args.max_steps is not None and train_loop.step >= args.max_steps:
                print(f"Reached max_steps ({args.max_steps}), stopping training")
                return False
            
            result = original_run_step(batch, cond)
            
            # Log test generation periodically
            if (args.log_test_generation and 
                train_loop.step > 0 and 
                train_loop.step % args.test_generation_interval == 0):
                train_loop.log_test_generation()
            
            return result
        
        train_loop.run_step = enhanced_run_step
        
        # Override run_loop to respect max_steps
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
    defaults = voxel_gaussian_model_and_diffusion_defaults()
    defaults.update(dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=3000000,  # 3M iterations for LR decay
        batch_size=8,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        class_cond=False,
        # Wandb specific
        wandb_project="voxel-gaussian-diffusion-clean",
        wandb_run="",
        log_test_generation=True,
        test_generation_interval=500,
    ))
    
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    
    # Add voxel-specific arguments
    parser.add_argument(
        "--norm_stats_path", 
        type=str, 
        required=True,
        help="Path to global normalization statistics JSON file"
    )
    parser.add_argument(
        "--include_features", 
        nargs='+',
        default=["opacity", "scaling", "rotation", "features_dc", "features_rest"],
        choices=["opacity", "scaling", "rotation", "features_dc", "features_rest"],
        help="Gaussian features to include in training"
    )
    parser.add_argument(
        "--random_flip", 
        type=str2bool, 
        default=True,
        help="Apply random flipping for data augmentation"
    )
    parser.add_argument(
        "--random_rotate", 
        type=str2bool, 
        default=True,
        help="Apply random rotation for data augmentation"
    )
    parser.add_argument(
        "--beta_start", 
        type=float, 
        default=0.0015,
        help="Starting beta value for custom schedule"
    )
    parser.add_argument(
        "--beta_end", 
        type=float, 
        default=0.05,
        help="Ending beta value for custom schedule"
    )
    parser.add_argument(
        "--max_steps", 
        type=int, 
        default=None,
        help="Maximum number of training steps (for testing)"
    )
    
    return parser


if __name__ == "__main__":
    main() 