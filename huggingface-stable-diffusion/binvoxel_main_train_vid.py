#!/usr/bin/env python3
# Unconditional 3D DDPM on ShapeNetVox32 (e.g., chairs 03001627) with rich W&B logging:
# - losses, LR, grad norms
# - histograms (clean / noisy / predicted x0)
# - 2D slice grids (clean, noisy at t, predicted x0)
# - full denoising videos during eval
# - sample .binvox exports + PNG previews
#
# Requires:
# pip install torch torchvision accelerate diffusers wandb imageio tqdm huggingface_hub
# pip install git+https://github.com/dimatura/binvox-rw-py.git
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import glob
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
from PIL import Image
import imageio

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import wandb
from accelerate import Accelerator
from diffusers import DDPMScheduler
from diffusers.models import UNet3DConditionModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
import numpy as np
from skimage.measure import marching_cubes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


import trimesh


import binvox_rw


# ===============================
# Configuration
# ===============================
@dataclass
class TrainingConfig:
    # Data
    dataset_root: str = "/om/user/akiruga/datasets/ShapeNetVox32"
    class_id: str = "03001627"                 # chairs
    voxel_size: int = 32

    # Training
    train_batch_size: int = 8
    eval_batch_size: int = 8
    num_epochs: int = 50
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    num_workers: int = 4
    mixed_precision: str = "fp16"              # "no" | "fp16" | "bf16"

    # Diffusion
    num_train_timesteps: int = 1000
    num_inference_steps: int = 250

    # Logging / saving
    output_dir: str = "ddpm-vox-03001627-32"
    save_samples_every: int = 5
    save_model_every: int = 10
    eval_samples_to_generate: int = 16
    log_interval_steps: int = 400              # train-step interval for W&B visuals
    max_train_visuals_per_epoch: int = 8       # cap visuals per epoch to keep logs sane

    # Threshold for binarizing generated volumes (values in [-1, 1])
    binarize_threshold: float = 0.0

    # W&B
    wandb_project: str = "voxels-ddpm-3d"
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "online"                 # "online" | "offline" | "disabled"

    # Hub
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_private_repo: Optional[bool] = None

    # Reproducibility
    seed: int = 0


config = TrainingConfig()


# ===============================
# Utils
# ===============================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_binvox(volume_bool: np.ndarray, out_path: str) -> None:
    assert volume_bool.dtype == np.bool_, "Expected boolean array for binvox."
    dims = list(volume_bool.shape)
    vox = binvox_rw.Voxels(volume_bool, dims=dims, translate=[0.0, 0.0, 0.0], scale=1.0, axis_order="xyz")
    with open(out_path, "wb") as f:
        vox.write(f)


def volume_to_slice_grid(volume: np.ndarray, title: Optional[str] = None) -> Image.Image:
    """
    Build a tiled grid of orthogonal mid-slices from (D,H,W) float array in [-1,1] or [0,1].
    Returns a PIL grayscale image.
    """
    D, H, W = volume.shape
    mid_d, mid_h, mid_w = D // 2, H // 2, W // 2
    slices = [
        volume[mid_d, :, :],   # Z mid
        volume[:, mid_h, :],   # Y mid
        volume[:, :, mid_w],   # X mid
    ]
    tiles = []
    for sl in slices:
        sl = sl.copy()
        sl = (sl - sl.min()) / (sl.max() - sl.min() + 1e-8)
        sl = (sl * 255).astype(np.uint8)
        tiles.append(Image.fromarray(sl))

    # concat horizontally
    widths = [im.size[0] for im in tiles]
    heights = [im.size[1] for im in tiles]
    grid = Image.new("L", (sum(widths), max(heights)))
    x = 0
    for im in tiles:
        grid.paste(im, (x, 0))
        x += im.size[0]
    return grid


def tile_slices_for_preview(volume: np.ndarray, save_path: str) -> None:
    """
    Create a richer 3x5 tile of slices (5 per axis) for quick inspection.
    volume: float array (D,H,W) in [-1,1]
    """
    D, H, W = volume.shape
    mid_d, mid_h, mid_w = D // 2, H // 2, W // 2
    idxs = [mid_d - 4, mid_d - 2, mid_d, mid_d + 2, mid_d + 4]
    rows: List[Image.Image] = []

    for axis, indices in [("Z", idxs), ("Y", idxs), ("X", idxs)]:
        row_imgs = []
        for idx in indices:
            if axis == "Z":
                sl = volume[max(0, min(D - 1, idx)), :, :]
            elif axis == "Y":
                sl = volume[:, max(0, min(H - 1, idx)), :]
            else:
                sl = volume[:, :, max(0, min(W - 1, idx))]
            sl = (sl - sl.min()) / (sl.max() - sl.min() + 1e-8)
            row_imgs.append(Image.fromarray((sl * 255).astype(np.uint8)))
        widths = [im.size[0] for im in row_imgs]
        heights = [im.size[1] for im in row_imgs]
        row = Image.new("L", (sum(widths), max(heights)))
        x = 0
        for im in row_imgs:
            row.paste(im, (x, 0))
            x += im.size[0]
        rows.append(row)

    w = max(p.size[0] for p in rows)
    h = sum(p.size[1] for p in rows)
    grid = Image.new("L", (w, h))
    y = 0
    for p in rows:
        grid.paste(p, (0, y))
        y += p.size[1]
    grid.save(save_path)


def make_video_from_volumes(
    volumes: List[np.ndarray], out_path: str, fps: int = 12
) -> None:
    """
    volumes: list of (D,H,W) float arrays in [-1,1]; we convert each to a 3-slice grid frame.
    """
    frames = []
    for v in volumes:
        frame = volume_to_slice_grid(v)
        frames.append(np.array(frame))  # (H,W)
    frames = [np.repeat(f[..., None], 3, axis=-1) for f in frames]  # grayscale→RGB
    imageio.mimwrite(out_path, frames, fps=fps, macro_block_size=8)


# --- add this helper (place with other helpers) ---
def render_turntable_video_from_volume(
    volume_dhw: np.ndarray,
    out_path: str,
    threshold: float = 0.0,
    num_frames: int = 120,
    elev_deg: float = 25.0,
    spiral: bool = True,
    fps: int = 12,
) -> None:
    """
    Turntable video of a voxel volume via marching cubes + matplotlib (headless).
    volume_dhw: (D,H,W) float in [-1,1]
    """
    # binarize & mesh
    occ = (volume_dhw > threshold).astype(np.float32)
    if occ.max() <= 0:
        # empty → make a blank video
        h = w = occ.shape[1]
        imageio.mimwrite(out_path, [np.zeros((h, w, 3), np.uint8) for _ in range(num_frames)], fps=fps)
        return

    verts, faces, _, _ = marching_cubes(occ, level=0.5)
    verts -= verts.mean(0, keepdims=True)
    scale = np.abs(verts).max() + 1e-8
    verts /= scale

    # setup figure
    fig = plt.figure(figsize=(4, 4), dpi=256/4)  # ~256px frames
    ax = fig.add_subplot(111, projection="3d")
    tri = Poly3DCollection(verts[faces], linewidths=0.0, alpha=1.0)
    tri.set_facecolor((0.8, 0.8, 0.8, 1.0))
    ax.add_collection3d(tri)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_axis_off()

    frames = []
    for i in range(num_frames):
        azim = (360.0 * i) / num_frames
        elev = elev_deg + (10.0 * np.sin(2 * np.pi * i / num_frames)) if spiral else elev_deg
        ax.view_init(elev=elev, azim=azim)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        frames.append(frame.copy())

    plt.close(fig)
    imageio.mimwrite(out_path, frames, fps=fps, macro_block_size=8)


# ===============================
# Dataset
# ===============================
class BinvoxDataset(Dataset):
    """
    Loads ShapeNetVox32-style .binvox volumes (binary occupancy).
    Directory layout:
      <root>/<class_id>/<model_id>/model.binvox
    Returns {"images": tensor} where tensor is float32 in [-1,1], shape (1, D, H, W).
    """

    def __init__(self, root: str, class_id: str, voxel_size: int = 32, augment: bool = True) -> None:
        self.root = root
        self.class_id = class_id
        self.voxel_size = voxel_size
        self.augment = augment

        pattern = os.path.join(root, class_id, "*", "model.binvox")
        self.files = sorted(glob.glob(pattern))
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .binvox files found at {pattern}")

    def __len__(self) -> int:
        return len(self.files)

    def _read_binvox(self, path: str) -> np.ndarray:
        with open(path, "rb") as f:
            vox = binvox_rw.read_as_3d_array(f)
        vol = vox.data.astype(np.float32)  # (D,H,W), {0,1}
        if vol.shape != (self.voxel_size, self.voxel_size, self.voxel_size):
            raise ValueError(f"Expected ({self.voxel_size},{self.voxel_size},{self.voxel_size}), got {vol.shape}")
        return vol

    def _augment(self, vol: np.ndarray) -> np.ndarray:
        if random.random() < 0.5:
            vol = np.flip(vol, axis=0).copy()
        if random.random() < 0.5:
            vol = np.flip(vol, axis=1).copy()
        if random.random() < 0.5:
            vol = np.flip(vol, axis=2).copy()
        k = random.choice([0, 1, 2, 3])
        if k > 0:
            vol = np.rot90(vol, k=k, axes=(1, 2)).copy()
        return vol

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        vol = self._read_binvox(self.files[idx])
        if self.augment:
            vol = self._augment(vol)
        vol = vol * 2.0 - 1.0                     # {0,1} → [-1,1]
        return {"images": torch.from_numpy(vol)[None, ...]}


# ===============================
# Model
# ===============================
def build_unet3d(voxel_size: int) -> UNet3DConditionModel:
    return UNet3DConditionModel(
        sample_size=voxel_size,
        in_channels=1,
        out_channels=1,
        down_block_types=("DownBlock3D", "DownBlock3D", "DownBlock3D", "DownBlock3D"),
        up_block_types=("UpBlock3D", "UpBlock3D", "UpBlock3D", "UpBlock3D"),
        block_out_channels=(64, 128, 256, 256),
        layers_per_block=2,
        cross_attention_dim=None,
        norm_num_groups=16,
    )


# ===============================
# Inference helpers
# ===============================
@torch.no_grad()
def sample_volumes_with_video(
    model: UNet3DConditionModel,
    scheduler: DDPMScheduler,
    batch_size: int,
    voxel_size: int,
    num_inference_steps: int,
    device: torch.device,
    capture_video: bool = True,
) -> (torch.Tensor, Optional[List[List[np.ndarray]]]):
    """
    Reverse diffusion; optionally capture per-step volumes for video (only for the first item per batch to limit memory).
    Returns: (samples in [-1,1], per-sample list of step volumes or None)
    """
    scheduler.set_timesteps(num_inference_steps, device=device)
    x = torch.randn((batch_size, 1, voxel_size, voxel_size, voxel_size), device=device)

    per_sample_step_vols: Optional[List[List[np.ndarray]]] = [[] for _ in range(batch_size)] if capture_video else None

    # prepare unconditional conditioning (dummy encoder_hidden_states)
    cond_dim = getattr(model, "config", None)
    cond_dim = getattr(cond_dim, "cross_attention_dim", None) if cond_dim is not None else None
    cond_dim = cond_dim if cond_dim is not None else 1
    for i, t in enumerate(scheduler.timesteps):
        encoder_hidden_states = torch.zeros((batch_size, 1, cond_dim), device=device, dtype=x.dtype)
        noise_pred = model(x, t, encoder_hidden_states=encoder_hidden_states).sample
        out = scheduler.step(model_output=noise_pred, timestep=t, sample=x)
        x = out.prev_sample

        if capture_video:
            with torch.no_grad():
                for b in range(batch_size):
                    if len(per_sample_step_vols[b]) < num_inference_steps:  # record all steps
                        vol = x[b, 0].detach().clamp(-1, 1).cpu().numpy()
                        per_sample_step_vols[b].append(vol)

    x = x.clamp(-1.0, 1.0)
    return x, per_sample_step_vols


def predict_x0_from_eps(x_t: torch.Tensor, t: torch.LongTensor, eps: torch.Tensor, scheduler: DDPMScheduler) -> torch.Tensor:
    """
    x0 = (x_t - sqrt(1 - a_bar_t) * eps) / sqrt(a_bar_t)
    """
    a_bar = scheduler.alphas_cumprod.to(x_t.device)[t]             # (B,)
    a_bar = a_bar.view(-1, 1, 1, 1, 1)                             # broadcast
    x0 = (x_t - torch.sqrt(1 - a_bar) * eps) / torch.sqrt(a_bar + 1e-8)
    return x0.clamp(-1, 1)


# ===============================
# W&B logging helpers
# ===============================
def log_histograms(prefix: str, tensors: Dict[str, torch.Tensor], step: int):
    log_dict = {}
    for name, ten in tensors.items():
        arr = ten.detach().flatten().float().cpu().numpy()
        log_dict[f"{prefix}/{name}_hist"] = wandb.Histogram(arr)
        log_dict[f"{prefix}/{name}_min"] = float(arr.min()) if arr.size else 0.0
        log_dict[f"{prefix}/{name}_max"] = float(arr.max()) if arr.size else 0.0
        log_dict[f"{prefix}/{name}_mean"] = float(arr.mean()) if arr.size else 0.0
        log_dict[f"{prefix}/{name}_std"] = float(arr.std()) if arr.size else 0.0
        log_dict[f"{prefix}/{name}_sparsity_frac<=0"] = float((arr <= 0).mean()) if arr.size else 0.0
    wandb.log(log_dict, step=step)


def log_slice_images(prefix: str, volumes: Dict[str, torch.Tensor], step: int):
    images = {}
    for name, ten in volumes.items():
        vol = ten.detach().float().cpu().numpy()
        if vol.ndim == 5:    # (B,1,D,H,W) → use first
            vol = vol[0, 0]
        elif vol.ndim == 4:  # (1,D,H,W)
            vol = vol[0]
        grid = volume_to_slice_grid(vol)                    # PIL
        images[f"{prefix}/{name}_slices"] = wandb.Image(grid)
    if images:
        wandb.log(images, step=step)


def log_denoise_videos(prefix: str, per_sample_vols: List[List[np.ndarray]], step: int, fps: int = 12, max_items: int = 4):
    videos = {}
    for i, vols in enumerate(per_sample_vols[:max_items]):
        # convert list of (D,H,W) to array of (T, C, H, W)
        frames = []
        for v in vols:
            frame = volume_to_slice_grid(v)
            f = np.array(frame)  # (H,W)
            f = np.repeat(f[None, None, ...], 1, axis=0)   # (1,1,H,W)
            frames.append(f)
        vid = np.concatenate(frames, axis=0)               # (T,1,H,W)
        videos[f"{prefix}/sample_{i}"] = wandb.Video(vid, fps=fps, format="mp4")
    if videos:
        wandb.log(videos, step=step)


def grad_global_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().data.norm(2).item() ** 2)
    return math.sqrt(total)


# ===============================
# Eval saving
# ===============================
def save_eval_samples(
    samples: torch.Tensor,
    output_dir: str,
    epoch: int,
    threshold: float,
) -> None:
    samples_np = samples.squeeze(1).detach().cpu().numpy()  # (B,D,H,W)
    out_samples_dir = os.path.join(output_dir, "samples")
    ensure_dir(out_samples_dir)

    for i, vol in enumerate(samples_np):
        vol_bin = (vol > threshold)
        binvox_path = os.path.join(out_samples_dir, f"epoch{epoch:04d}_sample{i:03d}.binvox")
        save_binvox(vol_bin.astype(np.bool_), binvox_path)
        preview_path = os.path.join(out_samples_dir, f"epoch{epoch:04d}_sample{i:03d}_preview.png")
        tile_slices_for_preview(vol, preview_path)


# ===============================
# Evaluate
# ===============================
def evaluate_and_log(
    accelerator: Accelerator,
    config: TrainingConfig,
    epoch: int,
    model: UNet3DConditionModel,
    scheduler: DDPMScheduler,
) -> None:
    device = accelerator.device
    total = config.eval_samples_to_generate
    bs = config.eval_batch_size
    num_batches = math.ceil(total / bs)
    all_samples: List[torch.Tensor] = []
    videos_captured: List[List[np.ndarray]] = []

    model.eval()
    for b in range(num_batches):
        cur_bs = bs if (b + 1) * bs <= total else total - b * bs
        samples, per_sample_vols = sample_volumes_with_video(
            model=accelerator.unwrap_model(model),
            scheduler=scheduler,
            batch_size=cur_bs,
            voxel_size=config.voxel_size,
            num_inference_steps=config.num_inference_steps,
            device=device,
            capture_video=True,
        )
        all_samples.append(samples)
        if per_sample_vols is not None:
            videos_captured.extend(per_sample_vols)

    all_samples_t = torch.cat(all_samples, dim=0)
    save_eval_samples(all_samples_t, config.output_dir, epoch, threshold=config.binarize_threshold)
    turn_dir = os.path.join(config.output_dir, "turntables")
    ensure_dir(turn_dir)

    turn_videos = {}
    max_items = min(4, all_samples_t.shape[0])
    for i in range(max_items):
        vol = all_samples_t[i, 0].detach().cpu().numpy()
        vid_path = os.path.join(turn_dir, f"epoch{epoch:04d}_sample{i:03d}_turntable.mp4")
        render_turntable_video_from_volume(
            volume_dhw=vol,
            out_path=vid_path,
            threshold=config.binarize_threshold,
            num_frames=120,
            elev_deg=25.0,
            spiral=True,
            fps=12,
        )
        turn_videos[f"eval/turntable_sample_{i}"] = wandb.Video(vid_path)

    wandb.log(turn_videos, step=epoch)

    # W&B: slice previews of first few samples
    preview_dict = {f"eval/sample_{i}": all_samples_t[i:i+1] for i in range(min(4, all_samples_t.shape[0]))}
    log_slice_images("eval", preview_dict, step=epoch)

    # W&B: denoising videos
    log_denoise_videos("eval/denoise_video", videos_captured, step=epoch, fps=12, max_items=4)

    model.train()


# ===============================
# Train
# ===============================
def train() -> None:
    set_seed(config.seed)
    ensure_dir(config.output_dir)

    # Accelerator
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with=None,  # We'll use WandB directly
        project_dir=os.path.join(config.output_dir, "logs"),
    )

    # W&B init (main process only)
    if accelerator.is_main_process:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            mode=config.wandb_mode,
            config={
                **vars(config),
                "task": "unconditional_3d_ddpm",
                "data.class_id": config.class_id,
                "data.root": config.dataset_root,
            },
        )

    # Dataset / loader
    dataset = BinvoxDataset(
        root=config.dataset_root,
        class_id=config.class_id,
        voxel_size=config.voxel_size,
        augment=True,
    )
    train_dataloader = DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Model / scheduler / optimizer
    model = build_unet3d(voxel_size=config.voxel_size)
    noise_scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps, prediction_type="epsilon")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    # Prepare
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0
    visuals_logged_this_epoch = 0

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(
            total=len(train_dataloader),
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch}",
        )
        visuals_logged_this_epoch = 0

        for step, batch in enumerate(train_dataloader):
            if accelerator.is_main_process:
                evaluate_and_log(accelerator, config, epoch, model, noise_scheduler)
                exit()
            clean_vols: torch.Tensor = batch["images"].to(accelerator.device)  # (B,1,D,H,W) in [-1,1]
            noise = torch.randn_like(clean_vols)
            bs = clean_vols.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,),
                device=clean_vols.device, dtype=torch.int64
            )
            noisy = noise_scheduler.add_noise(clean_vols, noise, timesteps)

            with accelerator.accumulate(model):
                # unconditional dummy encoder_hidden_states
                cond_dim = getattr(model, "config", None)
                cond_dim = getattr(cond_dim, "cross_attention_dim", None) if cond_dim is not None else None
                cond_dim = cond_dim if cond_dim is not None else 1
                encoder_hidden_states = torch.zeros((bs, 1, cond_dim), device=noisy.device, dtype=noisy.dtype)
                noise_pred = model(noisy, timesteps, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Metrics
            if accelerator.is_main_process:
                log_payload = {
                    "train/loss": loss.detach().float().item(),
                    "train/lr": lr_scheduler.get_last_lr()[0],
                    "train/step": global_step,
                    "train/epoch": epoch,
                }
                try:
                    log_payload["train/grad_global_norm"] = grad_global_norm(accelerator.unwrap_model(model))
                except Exception:
                    pass
                wandb.log(log_payload, step=global_step)

            # Visuals at interval
            should_log_visual = (
                accelerator.is_main_process
                and (global_step % config.log_interval_steps == 0)
                and (visuals_logged_this_epoch < config.max_train_visuals_per_epoch)
            )

            if should_log_visual:
                # Predicted x0 from epsilon
                with torch.no_grad():
                    x0_pred = predict_x0_from_eps(noisy, timesteps, noise_pred, noise_scheduler)

                # Histograms
                log_histograms(
                    "train",
                    {
                        "clean": clean_vols[0, 0],
                        "noisy": noisy[0, 0],
                        "pred_eps": noise_pred[0, 0],
                        "pred_x0": x0_pred[0, 0],
                    },
                    step=global_step,
                )

                # Slice images (grids)
                log_slice_images(
                    "train",
                    {
                        "clean": clean_vols[0:1],
                        "noisy": noisy[0:1],
                        "pred_x0": x0_pred[0:1],
                    },
                    step=global_step,
                )

                visuals_logged_this_epoch += 1

            progress_bar.update(1)
            global_step += 1

        # Eval + save
        if accelerator.is_main_process:
            if (epoch % config.save_samples_every == 0) or (epoch == config.num_epochs - 1):
                evaluate_and_log(accelerator, config, epoch, model, noise_scheduler)

            if (epoch % config.save_model_every == 0) or (epoch == config.num_epochs - 1):
                # Save locally
                unwrapped = accelerator.unwrap_model(model)
                model_dir = os.path.join(config.output_dir, "unet")
                sched_dir = os.path.join(config.output_dir, "scheduler")
                ensure_dir(model_dir)
                ensure_dir(sched_dir)
                unwrapped.save_pretrained(model_dir)
                noise_scheduler.save_pretrained(sched_dir)

                # Optionally push entire folder (including samples) to Hub
                if config.push_to_hub:
                    repo_id = create_repo(
                        repo_id=config.hub_model_id or Path(config.output_dir).name,
                        exist_ok=True,
                        private=config.hub_private_repo,
                    ).repo_id
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=config.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["**/*.pt", "**/*.pth"],
                    )

    if accelerator.is_main_process:
        wandb.finish()

    accelerator.end_training()


if __name__ == "__main__":
    train()
