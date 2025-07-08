#!/usr/bin/env python3
"""
Train a 3D diffusion model on voxel gaussian data with comprehensive wandb logging.

Features complete visualization pipeline including:
- train/noisy_voxel_viz: Input (noisy) voxels rendered as spinning video + multiview
- train/after_noise_voxel_viz: Denoised voxels at training time
- test/noise_grid: Complete noise at test time  
- test/denoised_grid_generation: Full denoising from noise
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
from guided_diffusion.voxel_gaussian_datasets import load_voxel_gaussian_data, denormalize_gaussian_features
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


class GaussianVoxelGrid(th.nn.Module):
    """
    Lightweight voxel grid class for rendering, adapted from splatter-image.
    """
    def __init__(self, grid_size=32, max_sh_degree=3, device="cuda"):
        super().__init__()
        self.grid_size = grid_size
        self.max_sh_degree = max_sh_degree
        
        # Create fixed voxel centers (using same coordinate system as training data)
        half_extent = 0.5
        coords = th.linspace(-half_extent, half_extent, grid_size, device=device)
        x, y, z = th.meshgrid(coords, coords, coords, indexing="ij")
        voxel_centers = th.stack([x, y, z], dim=-1).reshape(grid_size, grid_size, grid_size, 3)
        self.register_buffer("voxel_centers", voxel_centers)
        
        # Placeholders for gaussian parameters
        self.opacity = th.nn.Parameter(th.zeros(grid_size, grid_size, grid_size, device=device))
        self.scaling = th.nn.Parameter(th.zeros(grid_size, grid_size, grid_size, 3, device=device))
        self.rotation = th.nn.Parameter(th.zeros(grid_size, grid_size, grid_size, 4, device=device))
        self.features_dc = th.nn.Parameter(th.zeros(grid_size, grid_size, grid_size, 3, device=device))
        if max_sh_degree > 0:
            sh_dim = (max_sh_degree + 1) ** 2 - 1
            self.features_rest = th.nn.Parameter(th.zeros(grid_size, grid_size, grid_size, sh_dim, 3, device=device))
        else:
            self.register_buffer("features_rest", th.zeros(grid_size, grid_size, grid_size, 0, 3, device=device))

    def load_from_volume(self, volume, include_features, denormalize=True, norm_stats=None):
        """
        Load parameters from a 56-channel volume tensor.
        
        Args:
            volume: [C, D, H, W] tensor with C=56 channels
            include_features: List of feature names
            denormalize: Whether to denormalize the features
            norm_stats: Normalization statistics for proper denormalization
        """
        if denormalize:
            volume = denormalize_gaussian_features(volume.detach(), include_features, norm_stats)
        
        channel_idx = 0
        
        for feature_name in include_features:
            if feature_name == "opacity":
                # [1, D, H, W] -> [D, H, W]
                self.opacity.data.copy_(volume[channel_idx])
                channel_idx += 1
                
            elif feature_name == "scaling":
                # [3, D, H, W] -> [D, H, W, 3]
                self.scaling.data.copy_(volume[channel_idx:channel_idx+3].permute(1, 2, 3, 0))
                channel_idx += 3
                
            elif feature_name == "rotation":
                # [4, D, H, W] -> [D, H, W, 4]
                self.rotation.data.copy_(volume[channel_idx:channel_idx+4].permute(1, 2, 3, 0))
                channel_idx += 4
                
            elif feature_name == "features_dc":
                # [3, D, H, W] -> [D, H, W, 3]
                self.features_dc.data.copy_(volume[channel_idx:channel_idx+3].permute(1, 2, 3, 0))
                channel_idx += 3
                
            elif feature_name == "features_rest":
                # [45, D, H, W] -> [D, H, W, 15, 3]
                rest_channels = volume[channel_idx:channel_idx+45]  # [45, D, H, W]
                rest_reshaped = rest_channels.permute(1, 2, 3, 0).reshape(self.grid_size, self.grid_size, self.grid_size, 15, 3)
                self.features_rest.data.copy_(rest_reshaped)
                channel_idx += 45

    def get_gaussian_parameters_at_voxels(self, opacity_threshold=0.01):
        """Convert voxel grid to gaussian parameters for rendering."""
        pos = self.voxel_centers.reshape(-1, 3)
        alpha = th.sigmoid(self.opacity.reshape(-1))
        mask = alpha > opacity_threshold
        
        print(f"  Opacity stats: min={alpha.min().item():.6f}, max={alpha.max().item():.6f}, mean={alpha.mean().item():.6f}")
        print(f"  Gaussians above threshold ({opacity_threshold}): {mask.sum().item()}/{mask.numel()}")
        
        if mask.sum() == 0:
            print(f"  No gaussians above threshold! Using threshold=0.0")
            mask = alpha > 0.0
            if mask.sum() == 0:
                print(f"  Still no gaussians! Keeping first gaussian")
                mask[0] = True
        
        # Check scaling values before exp
        scaling_raw = self.scaling.reshape(-1, 3)[mask]
        print(f"  Scaling (raw): min={scaling_raw.min().item():.4f}, max={scaling_raw.max().item():.4f}")
        scaling_exp = th.exp(scaling_raw)
        print(f"  Scaling (exp): min={scaling_exp.min().item():.4f}, max={scaling_exp.max().item():.4f}")
        
        # Check rotation normalization
        rotation_raw = self.rotation.reshape(-1, 4)[mask]
        rotation_norm = F.normalize(rotation_raw, dim=-1)
        print(f"  Rotation norm: mean={th.norm(rotation_norm, dim=-1).mean().item():.6f}")
        
        sh = self.features_rest
        result = {
            "xyz": pos[mask],
            "opacity": alpha[mask, None],
            "scaling": scaling_exp,
            "rotation": rotation_norm,
            "features_dc": self.features_dc.reshape(-1, 3)[mask, None],
            "features_rest": sh.reshape(-1, sh.shape[-2], 3)[mask] if self.max_sh_degree > 0 
                           else th.zeros(mask.sum(), 0, 3, device=self.opacity.device)
        }
        
        print(f"  Final gaussian count: {result['xyz'].shape[0]}")
        return result


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


def render_voxel_views(voxel_grid, dataset_sample, cfg, device, num_views=8, opacity_threshold=0.01):
    """
    Render multiple views of a voxel grid.
    
    Args:
        voxel_grid: GaussianVoxelGrid instance
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
    
    voxel_grid.eval()
    bg_color = [1, 1, 1] if cfg.data.white_background else [0, 0, 0]
    background = th.tensor(bg_color, dtype=th.float32, device=device)
    
    # Get gaussian parameters
    gauss = voxel_grid.get_gaussian_parameters_at_voxels(opacity_threshold)
    
    # Select views to render
    total_views = dataset_sample['gt_images'].shape[0] if 'gt_images' in dataset_sample else 25
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


def save_video_frames(frames, output_path, fps=10):
    """Save frames as MP4 video."""
    if frames and len(frames) > 0:
        imageio.mimsave(output_path, frames, fps=fps)
        return output_path
    return None


class VoxelGaussianWandbTrainLoop(TrainLoop):
    """
    Custom training loop with wandb logging and voxel rendering.
    """
    
    def __init__(self, *args, include_features=None, global_dataset=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.include_features = include_features or ["opacity", "scaling", "rotation", "features_dc", "features_rest"]
        self.alpha_bar_tensor = th.tensor(self.diffusion.alphas_cumprod, device=dist_util.dev())
        self.render_config = create_render_config()
        
        # Get global normalization stats from the provided dataset
        self.global_norm_stats = None
        if global_dataset is not None and hasattr(global_dataset, 'global_norm_stats'):
            self.global_norm_stats = global_dataset.global_norm_stats
            print(f"Retrieved global normalization stats: {len(self.global_norm_stats)} parameters")
        else:
            print("Warning: No global dataset provided or missing normalization stats")
        
        # Try to get dataset sample for camera info
        self.dataset_sample = None
        try:
            # Get a sample from the data loader for camera information
            sample_batch, sample_cond = next(iter(self.data))
            if len(sample_batch) > 0:
                print("Warning: Using dummy camera setup for rendering")
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
                    self.log_noisy_voxel_viz(x_t[0:1], f"train/noisy_voxel_viz_step_{self.step}")
                
                terms = {}
                model_output = self.ddp_model(x_t, t, **micro_cond)
                
                # Log denoised output visualization occasionally  
                if self.step % self.log_interval == 0 and self.step % 50 == 0:
                    self.log_denoised_voxel_viz(model_output[0:1], f"train/after_noise_voxel_viz_step_{self.step}")
                
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
    
    def log_noisy_voxel_viz(self, noisy_volume, prefix):
        """Log visualization of noisy voxel input."""
        try:
            # Create voxel grid from volume
            voxel_grid = GaussianVoxelGrid(device=dist_util.dev())
            # For noisy input, don't denormalize (it's already normalized for training)
            voxel_grid.load_from_volume(noisy_volume[0], self.include_features, denormalize=False)
            
            # Render views
            frames = render_voxel_views(voxel_grid, self.dataset_sample or {}, self.render_config, dist_util.dev())
            
            if frames:
                # Create multiview grid
                multiview_grid = create_multiview_grid(frames)
                
                # Log to wandb
                wandb.log({
                    f"{prefix}_multiview": wandb.Image(multiview_grid),
                    f"{prefix}_video": wandb.Video(np.array(frames), fps=5, format="mp4")
                })
                
        except Exception as e:
            print(f"Warning: Failed to log noisy voxel viz: {e}")
    
    def log_denoised_voxel_viz(self, denoised_volume, prefix):
        """Log visualization of denoised voxel output."""
        try:
            print(f"\n=== DEBUGGING {prefix} ===")
            
            # Check the input volume stats
            vol = denoised_volume[0]
            print(f"Input volume shape: {vol.shape}")
            print(f"Input volume range: [{vol.min().item():.4f}, {vol.max().item():.4f}]")
            print(f"Input volume mean: {vol.mean().item():.4f}, std: {vol.std().item():.4f}")
            
            # Check normalization stats
            if self.global_norm_stats:
                print(f"Global norm stats available: {len(self.global_norm_stats)} parameters")
                for key, val in list(self.global_norm_stats.items())[:10]:  # Print first 10
                    print(f"  {key}: {val}")
            else:
                print("WARNING: No global normalization stats!")
            
            # Create voxel grid from volume
            voxel_grid = GaussianVoxelGrid(device=dist_util.dev())
            # Use global normalization stats for proper denormalization
            voxel_grid.load_from_volume(vol, self.include_features, denormalize=True, norm_stats=self.global_norm_stats)
            
            # Check gaussian parameters after denormalization
            gauss_params = voxel_grid.get_gaussian_parameters_at_voxels(opacity_threshold=0.01)
            print(f"\nGaussian parameters after denormalization:")
            for key, val in gauss_params.items():
                if isinstance(val, th.Tensor):
                    print(f"  {key}: shape={val.shape}, range=[{val.min().item():.4f}, {val.max().item():.4f}], mean={val.mean().item():.4f}")
            
            # Render views
            frames = render_voxel_views(voxel_grid, self.dataset_sample or {}, self.render_config, dist_util.dev())
            
            # Save individual frames for debugging
            import os
            debug_dir = f"debug_frames_{prefix.replace('/', '_')}_step_{self.step}"
            os.makedirs(debug_dir, exist_ok=True)
            
            if frames:
                print(f"Rendered {len(frames)} frames")
                for i, frame in enumerate(frames):
                    print(f"  Frame {i}: shape={frame.shape}, range=[{frame.min()}, {frame.max()}], mean={frame.mean():.4f}")
                    # Save frame
                    try:
                        from PIL import Image
                        img = Image.fromarray(frame)
                        img.save(f"{debug_dir}/frame_{i:02d}.png")
                    except ImportError:
                        print(f"    PIL not available, skipping frame save")
                print(f"Saved frames to {debug_dir}/")
                
                # Create multiview grid
                multiview_grid = create_multiview_grid(frames)
                print(f"Multiview grid: shape={multiview_grid.shape}, range=[{multiview_grid.min()}, {multiview_grid.max()}]")
                
                # Save multiview grid
                try:
                    from PIL import Image
                    multiview_img = Image.fromarray(multiview_grid)
                    multiview_img.save(f"{debug_dir}/multiview_grid.png")
                except ImportError:
                    print("PIL not available, skipping multiview grid save")
                
                # Log to wandb
                wandb.log({
                    f"{prefix}_multiview": wandb.Image(multiview_grid),
                    f"{prefix}_video": wandb.Video(np.array(frames), fps=5, format="mp4")
                })
            else:
                print("No frames rendered!")
                
        except Exception as e:
            print(f"ERROR in log_denoised_voxel_viz: {e}")
            import traceback
            traceback.print_exc()
    
    def log_test_generation(self):
        """Log test-time generation from pure noise."""
        try:
            print("Generating test samples from pure noise...")
            
            # Sample from pure noise
            noise_shape = (1, 56, 32, 32, 32)  # [B, C, D, H, W]
            pure_noise = th.randn(noise_shape, device=dist_util.dev())
            
            # Log pure noise visualization
            voxel_grid_noise = GaussianVoxelGrid(device=dist_util.dev())
            # Pure noise doesn't need denormalization
            voxel_grid_noise.load_from_volume(pure_noise[0], self.include_features, denormalize=False)
            noise_frames = render_voxel_views(voxel_grid_noise, self.dataset_sample or {}, self.render_config, dist_util.dev())
            
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
            voxel_grid_denoised = GaussianVoxelGrid(device=dist_util.dev())
            # Use global normalization stats for proper denormalization
            voxel_grid_denoised.load_from_volume(denoised_sample[0], self.include_features, denormalize=True, norm_stats=self.global_norm_stats)
            denoised_frames = render_voxel_views(voxel_grid_denoised, self.dataset_sample or {}, self.render_config, dist_util.dev())
            
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
        tags=["voxel-gaussian", "3d-diffusion"]
    )
    
    # Log all training parameters
    wandb.config.update({
        "gaussian_features": args.include_features,
        "expected_channels": calculate_voxel_gaussian_channels(args.include_features),
        "grid_size": args.volume_size,
        "custom_loss_weighting": "alpha_bar_squared",
        "beta_schedule": f"linear_{args.beta_start}_to_{args.beta_end}",
    })

    dist_util.setup_dist()
    logger.configure()

    # Calculate expected channels
    expected_channels = calculate_voxel_gaussian_channels(args.include_features)
    logger.log(f"Expected input channels: {expected_channels}")
    logger.log(f"Included features: {args.include_features}")

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

    logger.log("creating voxel gaussian data loader...")
    data = load_voxel_gaussian_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        grid_size=args.volume_size,
        class_cond=args.class_cond,
        deterministic=False,
        random_flip=args.random_flip,
        random_rotate=args.random_rotate,
        include_features=args.include_features,
    )
    
    # Get the dataset object to access global normalization stats
    global_dataset = None
    try:
        # Create a temporary dataset to get the global stats
        from guided_diffusion.voxel_gaussian_datasets import VoxelGaussianDataset, _list_voxel_gaussian_dirs_recursively
        all_object_dirs = _list_voxel_gaussian_dirs_recursively(args.data_dir)
        global_dataset = VoxelGaussianDataset(
            args.volume_size,
            all_object_dirs,
            classes=None,
            shard=0,  # Use shard 0 to get stats from all data
            num_shards=1,  # Single shard to get complete stats
            random_flip=False,  # Don't apply augmentations for stats computation
            random_rotate=False,
            include_features=args.include_features,
        )
        logger.log(f"Loaded global dataset with {len(global_dataset.global_norm_stats)} normalization stats")
    except Exception as e:
        logger.log(f"Warning: Failed to create global dataset for stats: {e}")
        global_dataset = None

    logger.log("training voxel gaussian 3D diffusion model...")
    train_loop = VoxelGaussianWandbTrainLoop(
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
        global_dataset=global_dataset,
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
        wandb_project="voxel-gaussian-diffusion",
        wandb_run="",
        log_test_generation=True,
        test_generation_interval=500,
    ))
    
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    
    # Add voxel-specific arguments
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
    # wandb_project, wandb_run, log_test_generation, and test_generation_interval already defined in defaults above
    
    return parser


if __name__ == "__main__":
    main() 