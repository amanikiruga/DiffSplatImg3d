# Copyright 2021 Alex Yu

# First, install svox2
# Then, python opt.py <path_to>/nerf_synthetic/<scene> -t ckpt/<some_name>
# or use launching script:   sh launch.sh <EXP_NAME> <GPU> <DATA_DIR>
MAIN_DIR = "/om/user/akiruga/svox2/opt"
import sys 
sys.path.append(MAIN_DIR)
import torch
import torch.cuda
import torch.optim
import torch.nn.functional as F
import svox2
import json
import imageio
import os
from os import path
import shutil
import gc
import numpy as np
from pathlib import Path
import math
import argparse
import cv2
from util.dataset import datasets
from util.util import Timing, get_expon_lr_func, generate_dirs_equirect, viridis_cmap
from util import config_util

from warnings import warn
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from typing import NamedTuple, Optional, Union

device = "cuda" if torch.cuda.is_available() else "cpu"
# Create argument namespace object to mimic argparse
class Args:
    def __init__(self):
        pass

args = Args()

# Set all the default argument values as attributes
args.train_dir = '/om/user/akiruga/svox2/data/ckpts/shapenet_chairs_jupyter' # ckpts
# args.reso = "[[32, 32, 32]]"
# args.reso = "[[32, 32, 32]]"  # Using 32x32x32 with grid scaling for focused resolution
args.reso = "[[32, 32, 32]]"  # Using 32x32x32 with grid scaling for focused resolution
args.upsamp_every = 3 * 12800
args.init_iters = 0
args.upsample_density_add = 0.0
args.basis_type = 'sh'
args.basis_reso = 32
args.sh_dim = 9
args.mlp_posenc_size = 4
args.mlp_width = 32
args.background_nlayers = 0
args.background_reso = 512
args.n_iters = 10 * 12800
args.batch_size = 5000

# Grid scaling to focus 32x32x32 resolution on object region
# Current: object ~82% of image (105/128), Target: ~60% of image
# Scale factor: 60/82 ≈ 0.73 makes grid tighter around object
args.grid_scale_factor = 0.45  # Scale down scene radius to focus grid

args.sigma_optim = 'rmsprop'
args.lr_sigma = 3e1
args.lr_sigma_final = 5e-2
args.lr_sigma_decay_steps = 250000
args.lr_sigma_delay_steps = 15000
args.lr_sigma_delay_mult = 1e-2

args.sh_optim = 'rmsprop'
args.lr_sh = 1e-2
args.lr_sh_final = 5e-6
args.lr_sh_decay_steps = 250000
args.lr_sh_delay_steps = 0
args.lr_sh_delay_mult = 1e-2

args.lr_fg_begin_step = 0

args.bg_optim = 'rmsprop'
args.lr_sigma_bg = 3e0
args.lr_sigma_bg_final = 3e-3
args.lr_sigma_bg_decay_steps = 250000
args.lr_sigma_bg_delay_steps = 0
args.lr_sigma_bg_delay_mult = 1e-2

args.lr_color_bg = 1e-1
args.lr_color_bg_final = 5e-6
args.lr_color_bg_decay_steps = 250000
args.lr_color_bg_delay_steps = 0
args.lr_color_bg_delay_mult = 1e-2

args.basis_optim = 'rmsprop'
args.lr_basis = 1e-6
args.lr_basis_final = 1e-6
args.lr_basis_decay_steps = 250000
args.lr_basis_delay_steps = 0
args.lr_basis_begin_step = 0
args.lr_basis_delay_mult = 1e-2

args.rms_beta = 0.95
args.print_every = 20
args.save_every = 5
args.eval_every = 1

args.init_sigma = 0.1
args.init_sigma_bg = 0.1

args.log_mse_image = True
args.log_depth_map = True
args.log_depth_map_use_thresh = None

args.thresh_type = "weight"
args.weight_thresh = 0.0005 * 512
args.density_thresh = 5.0
args.background_density_thresh = 1.0+1e-9
args.max_grid_elements = 44_000_000

args.tune_mode = False
args.tune_nosave = False

args.lambda_tv = 1e-5
args.tv_sparsity = 0.01
args.tv_logalpha = False
args.lambda_tv_sh = 1e-3
args.tv_sh_sparsity = 0.01
args.lambda_tv_lumisphere = 0.0
args.tv_lumisphere_sparsity = 0.01
args.tv_lumisphere_dir_factor = 0.0
args.tv_decay = 1.0
args.lambda_l2_sh = 0.0
args.tv_early_only = 1
args.tv_contiguous = 1

args.lambda_sparsity = 0.0
args.lambda_beta = 0.0

args.lambda_tv_background_sigma = 1e-2
args.lambda_tv_background_color = 1e-2
args.tv_background_sparsity = 0.01

args.lambda_tv_basis = 0.0

args.weight_decay_sigma = 1.0
args.weight_decay_sh = 1.0

args.lr_decay = True
args.n_train = None
args.nosphereinit = True

# Add common args from config_util
args.data_dir = "/om/user/akiruga/datasets/srn_chairs_alternate_views/feab80af7f3e459120523e15ec10a342/viz"
# args.data_dir = "/weka/scratch/weka/tenenbaum/akiruga/svox2/data/datasets/lego_real_night_radial"
args.config = None
args.dataset_type = "shapenet" # "auto"
args.scene_scale = None
args.scale = None
args.seq_id = 1000
args.epoch_size = 12800
args.white_bkgd = True
args.llffhold = 8
args.normalize_by_bbox = False
args.data_bbox_scale = 1.2
args.cam_scale_factor = 0.95
args.normalize_by_camera = False
args.perm = False
args.step_size = 0.5 # 0.0125
args.sigma_thresh = 1e-8
args.stop_thresh = 1e-7
args.background_brightness = 1.0
args.renderer_backend = 'cuvol'
args.random_sigma_std = 0.0
args.random_sigma_std_background = 0.0
args.near_clip = 0.00
args.use_spheric_clip = False
args.enable_random = False
args.last_sample_opaque = False


# data specific args 
args.fov = 51.98948897809546
args.znear = 1.25
args.zfar = 2.75


# Calculate grid center and size using camera-aware approach (similar to GaussianVoxelGrid)
def calculate_grid_center_and_radius(cfg, camera_centers, cameras):
    """Calculate optimal grid center and radius based on camera geometry"""
    
    # Extract camera data like similarity_from_cameras does
    c2w = torch.stack([cam.c2w for cam in cameras]).cpu().numpy()  # Move to CPU first
    t = c2w[:, :3, 3]  # Camera positions
    R = c2w[:, :3, :3]  # Camera rotations
    
    # Get forward directions (where cameras are pointing) 
    fwds = np.sum(R * np.array([0, 0.0, 1.0]), axis=-1)
    
    # For each camera ray, find the closest point to the mean camera position
    # This finds where cameras are actually looking (the object center)
    mean_cam_pos = np.mean(t, axis=0)
    
    closest_points = []
    for i in range(len(t)):
        # For ray: P(s) = t[i] + s * fwds[i]
        # Find closest point on ray to mean_cam_pos
        ray_param = np.dot(mean_cam_pos - t[i], fwds[i])
        closest_point = t[i] + ray_param * fwds[i]
        closest_points.append(closest_point)
    
    # The object center is the median of these closest points (same as similarity_from_cameras)
    object_center = np.median(closest_points, axis=0)
    
    # Calculate grid dimensions based on FOV and z-range
    radius = np.linalg.norm(mean_cam_pos)
    fov_rad = np.deg2rad(cfg.fov)
    half_xy = radius * np.tan(0.5 * fov_rad) * 0.6  # 0.6 is a safety factor
    half_z = 0.3 * (cfg.zfar - cfg.znear)
    
    # Make it cubic (optional, or keep separate dimensions)
    half_extent = max(half_xy, half_z)
    
    return object_center, [half_extent, half_extent, half_extent]


# Add this function to save comparison images during eval
def save_comparison_image(grid, dset_test, epoch_id, args, device):
    """Save a single comparison image (rendered vs GT side by side)"""
    with torch.no_grad():
        # Use the first test image
        img_id = 0
        c2w = dset_test.c2w[img_id].to(device=device)
        cam = svox2.Camera(c2w,
                           dset_test.intrins.get('fx', img_id),
                           dset_test.intrins.get('fy', img_id),
                           dset_test.intrins.get('cx', img_id),
                           dset_test.intrins.get('cy', img_id),
                           width=dset_test.get_image_size(img_id)[1],
                           height=dset_test.get_image_size(img_id)[0],
                           ndc_coeffs=dset_test.ndc_coeffs)
        
        # Render the image
        rgb_pred = grid.volume_render_image(cam, use_kernel=True)
        rgb_gt = dset_test.gt[img_id].to(device=device)
        
        # Convert to numpy and scale to 0-255
        pred_img = (rgb_pred.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        gt_img = (rgb_gt.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        
        # Create side-by-side comparison (rendered on left, GT on right)
        comparison = np.concatenate([pred_img, gt_img], axis=1)
        
        # Save the comparison image
        comparison_path = Path(args.train_dir) / f"comparison_epoch_{epoch_id:04d}.png"
        imageio.imwrite(str(comparison_path), comparison)
        print(f"📸 Saved comparison image: {comparison_path}")
        
        return comparison_path



def get_reference_grid(device="cuda"): 
    os.makedirs(args.train_dir, exist_ok=True)
    summary_writer = SummaryWriter(args.train_dir)

    reso_list = json.loads(args.reso)
    reso_id = 0

    with open(path.join(args.train_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        # Changed name to prevent errors
        # shutil.copyfile(__file__, path.join(args.train_dir, 'opt_frozen.py'))

    torch.manual_seed(20200823)
    np.random.seed(20200823)

    factor = 1
    dset = datasets[args.dataset_type](
                args.data_dir,
                split="train",
                device=device, # Amani code: use device
                factor=factor,
                n_images=args.n_train,
                **config_util.build_data_options(args))

    if args.background_nlayers > 0 and not dset.should_use_background:
        warn('Using a background model for dataset type ' + str(type(dset)) + ' which typically does not use background')

    dset_test = datasets[args.dataset_type](
            args.data_dir, split="test", **config_util.build_data_options(args))

    global_start_time = datetime.now()

    # Apply grid scaling to focus 32x32x32 resolution on object region
    # Scale down scene radius to make grid tighter around object
    # Handle scene_radius as tensor/array for proper scaling
    if isinstance(dset.scene_radius, (list, tuple)):
        scaled_scene_radius = [r * args.grid_scale_factor for r in dset.scene_radius]
    else:
        # If it's already a tensor
        scaled_scene_radius = dset.scene_radius * args.grid_scale_factor

    # ------------------------------------------------------------
    # Amani code 

    # Use FOV to calculate proper intrinsics for 128x128 images
    fov_degrees = 51.98948897809546  # From your config
    fov_radians = fov_degrees * np.pi / 180.0
    image_size = 128  # Your actual image size

    # Calculate focal length from FOV: focal = (width/2) / tan(fov/2)
    focal_length = (image_size / 2.0) / np.tan(fov_radians / 2.0)
    principal_point = image_size / 2.0  # Center of image

    print(f"  FOV: {fov_degrees} degrees = {fov_radians} radians")
    print(f"  Image size: {image_size}x{image_size}")
    print(f"  Calculated focal length: {focal_length}")
    print(f"  Principal point: ({principal_point}, {principal_point})")

    # Override the intrinsics with FOV-calculated values
    from util.util import Intrin
    dset.intrins_full = Intrin(focal_length, focal_length, principal_point, principal_point)
    dset.intrins = dset.intrins_full

    dset_test.intrins_full = dset.intrins_full
    dset_test.intrins = dset_test.intrins_full


    # ------------------------------------------------------------


    resample_cameras = [
            svox2.Camera(c2w.to(device=device), # Amani code: use device
                        dset.intrins.get('fx', i),
                        dset.intrins.get('fy', i),
                        dset.intrins.get('cx', i),
                        dset.intrins.get('cy', i),
                        width=dset.get_image_size(i)[1],
                        height=dset.get_image_size(i)[0],
                        ndc_coeffs=dset.ndc_coeffs) for i, c2w in enumerate(dset.c2w)
        ]

    # ------------------------------------------------------------
    # Amani code 
    # Extract camera centers from your cameras
    camera_centers = torch.stack([cam.c2w[:3, 3] for cam in resample_cameras])

    # Calculate optimal center and radius
    optimal_center, optimal_radius = calculate_grid_center_and_radius(
        args, camera_centers, resample_cameras)

    # custom_center = [0.0,0.0, (args.zfar + args.znear)/2]
    custom_center = [0.0,0.0,0.0]

    print(f"Calculated optimal center: {optimal_center}")
    print(f"Calculated optimal radius: {optimal_radius}")
    print(f"Custom center: {custom_center}")
    print(f"Original dataset center: {dset.scene_center}")

    print(f"Original scene radius: {dset.scene_radius}")
    print(f"Scaled scene radius: {scaled_scene_radius} (scale factor: {args.grid_scale_factor})")

    # Apply your scale factor to the calculated radius
    scaled_optimal_radius = [r * args.grid_scale_factor for r in optimal_radius]


    # ------------------------------------------------------------






    grid = svox2.SparseGrid(reso=reso_list[reso_id],
                            # center=dset.scene_center,
                            center=custom_center, # Amani code: use custom center
                            radius=scaled_scene_radius,
                            use_sphere_bound=dset.use_sphere_bound and not args.nosphereinit,
                            # use_sphere_bound=False, # Amani code: use sphere bound
                            basis_dim=args.sh_dim,
                            use_z_order=True,
                            device=device, # Amani code: use device to set the device for the grid      
                            basis_reso=args.basis_reso,
                            basis_type=svox2.__dict__['BASIS_TYPE_' + args.basis_type.upper()],
                            mlp_posenc_size=args.mlp_posenc_size,
                            mlp_width=args.mlp_width,
                            background_nlayers=args.background_nlayers,
                            background_reso=args.background_reso)

    # DC -> gray; mind the SH scaling!
    grid.sh_data.data[:] = 0.0
    grid.density_data.data[:] = 0.0 if args.lr_fg_begin_step > 0 else args.init_sigma

    if grid.use_background:
        grid.background_data.data[..., -1] = args.init_sigma_bg
        #  grid.background_data.data[..., :-1] = 0.5 / svox2.utils.SH_C0

    #  grid.sh_data.data[:, 0] = 4.0
    #  osh = grid.density_data.data.shape
    #  den = grid.density_data.data.view(grid.links.shape)
    #  #  den[:] = 0.00
    #  #  den[:, :256, :] = 1e9
    #  #  den[:, :, 0] = 1e9
    #  grid.density_data.data = den.view(osh)

    optim_basis_mlp = None

    if grid.basis_type == svox2.BASIS_TYPE_3D_TEXTURE:
        grid.reinit_learned_bases(init_type='sh')
        #  grid.reinit_learned_bases(init_type='fourier')
        #  grid.reinit_learned_bases(init_type='sg', upper_hemi=True)
        #  grid.basis_data.data.normal_(mean=0.28209479177387814, std=0.001)

    elif grid.basis_type == svox2.BASIS_TYPE_MLP:
        # MLP!
        optim_basis_mlp = torch.optim.Adam(
                        grid.basis_mlp.parameters(),
                        lr=args.lr_basis
                    )


    grid.requires_grad_(True)
    config_util.setup_render_opts(grid.opt, args)
    print('Render options', grid.opt)

    gstep_id_base = 0

    ckpt_path = path.join(args.train_dir, 'ckpt.npz')

    lr_sigma_func = get_expon_lr_func(args.lr_sigma, args.lr_sigma_final, args.lr_sigma_delay_steps,
                                    args.lr_sigma_delay_mult, args.lr_sigma_decay_steps)
    lr_sh_func = get_expon_lr_func(args.lr_sh, args.lr_sh_final, args.lr_sh_delay_steps,
                                args.lr_sh_delay_mult, args.lr_sh_decay_steps)
    lr_basis_func = get_expon_lr_func(args.lr_basis, args.lr_basis_final, args.lr_basis_delay_steps,
                                args.lr_basis_delay_mult, args.lr_basis_decay_steps)
    lr_sigma_bg_func = get_expon_lr_func(args.lr_sigma_bg, args.lr_sigma_bg_final, args.lr_sigma_bg_delay_steps,
                                args.lr_sigma_bg_delay_mult, args.lr_sigma_bg_decay_steps)
    lr_color_bg_func = get_expon_lr_func(args.lr_color_bg, args.lr_color_bg_final, args.lr_color_bg_delay_steps,
                                args.lr_color_bg_delay_mult, args.lr_color_bg_decay_steps)
    lr_sigma_factor = 1.0
    lr_sh_factor = 1.0
    lr_basis_factor = 1.0

    last_upsamp_step = args.init_iters

    if args.enable_random:
        warn("Randomness is enabled for training (normal for LLFF & scenes with background)")

    epoch_id = -1

    # first_vid_path = Path(args.train_dir) / "video_00000.mp4"
    # render_video(grid,
    #              resample_cameras[:60],   # 60 poses ≈ 5 s @ 12 fps
    #              first_vid_path,
    #              fps=12, crop=1) 

    # Add gray density to visualize focused grid bounds in the first video
    print("Setting temporary gray density to visualize grid bounds...")
    grid.density_data.data[:] = 100.0  # High density for complete opacity
    grid.sh_data.data[:] = -0.5        # Negative to counteract +0.5 offset in rendering for black color


    # Re-render first video with visible grid bounds
    first_vid_path_bounds = Path(args.train_dir) / "video_00000_bounds.mp4" 
    render_video(grid, resample_cameras, first_vid_path_bounds, fps=12, crop=1)

    # Reset density to proper initial values for training
    grid.density_data.data[:] = 0.0 if args.lr_fg_begin_step > 0 else args.init_sigma
    grid.sh_data.data[:] = 0.0  # Reset SH coefficients

    print(f"✅ Grid bounds video saved! Density reset to {args.init_sigma if args.lr_fg_begin_step == 0 else 0.0} for training")
    
    return grid, resample_cameras



def render_video(grid,
                 cameras,
                 out_path,
                 fps: int = 12,
                 crop: float = 1.0):
    """
    Render a simple orbit video of the current grid and dump to MP4.
    Args:
        grid (svox2.SparseGrid): trained / training grid
        cameras (List[svox2.Camera]): list of camera poses
        out_path (str | Path): where to write the mp4
        fps (int): frames per second
        crop (float): 1.0 = full res, <1 crops center
    """
    grid.eval()                        # just for safety
    frames = []
    with torch.no_grad():
        for cam in cameras:
            # Optional center‑crop so you can render faster mid‑training
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
    imageio.mimwrite(str(out_path), frames, fps=fps, macro_block_size=8)
    print(f"✔️  Saved preview video → {out_path}")
    grid.train()
