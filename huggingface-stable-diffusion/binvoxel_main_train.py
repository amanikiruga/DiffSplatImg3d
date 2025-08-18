#!/usr/bin/env python3
# Unconditional 3D DDPM on ShapeNetVox32 (e.g., chairs 03001627) with rich W&B logging.
# Logs: loss/LR/grad-norm + histograms (clean/noisy/pred_x0) + single-frame voxel renders:
#   train clean, train noisy@t, pred x0, and unconditional test generation.
#
# Requires:
# pip install torch torchvision accelerate diffusers wandb tqdm huggingface_hub
# pip install git+https://github.com/dimatura/binvox-rw-py.git

import os
import glob
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

import binvox_rw
import argparse


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
    num_epochs: int = 50000
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
    log_interval_steps: int = 400
    max_train_visuals_per_epoch: int = 8

    # Threshold for binarizing / rendering (volumes are in [-1, 1])
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


def vox_frame(vol_dhw: np.ndarray, thr: float = 0.0) -> np.ndarray:
    """
    Solid voxel snapshot via matplotlib's ax.voxels (fast enough for 32^3).
    Returns uint8 RGB image array.
    """
    # rot90
    vol_dhw = np.rot90(vol_dhw, k=1, axes=(1, 2))
    occ = (vol_dhw > thr)
    fig = plt.figure(figsize=(3, 3), dpi=256 // 3)
    ax = fig.add_subplot(111, projection="3d")
    ax.voxels(occ, facecolors="#888888", edgecolor="black", linewidth=0.1)
    ax.set_axis_off()
    ax.view_init(25, 45)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.tostring_rgb(), np.uint8).reshape(h, w, 3).copy()
    plt.close(fig)
    return img


# ===============================
# Dataset
# ===============================
class BinvoxDataset(Dataset):
    """
    Loads ShapeNetVox32-style .binvox volumes (binary occupancy).
    <root>/<class_id>/<model_id>/model.binvox
    Returns {"images": tensor} float32 in [-1,1], shape (1, D, H, W).
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

def get_attention_hidden_states(model: UNet3DConditionModel, batch_size: int, device: torch.device) -> torch.Tensor:
    cond_dim = getattr(model, "config", None)
    cond_dim = getattr(cond_dim, "cross_attention_dim", None) if cond_dim is not None else None
    cond_dim = cond_dim if cond_dim is not None else 1
    return torch.zeros((batch_size, 1, cond_dim), device=device, dtype=torch.float32)

@torch.no_grad()
def sample_volumes(
    model: UNet3DConditionModel,
    scheduler: DDPMScheduler,
    batch_size: int,
    voxel_size: int,
    num_inference_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Reverse diffusion to sample batches of volumes in [-1,1]."""
    scheduler.set_timesteps(num_inference_steps, device=device)
    x = torch.randn((batch_size, 1, voxel_size, voxel_size, voxel_size), device=device)
    # prepare unconditional conditioning (dummy encoder_hidden_states)
    encoder_hidden_states = get_attention_hidden_states(model, batch_size, device)
    for t in scheduler.timesteps:
        eps = model(x, t, encoder_hidden_states=encoder_hidden_states).sample
        x = scheduler.step(model_output=eps, timestep=t, sample=x).prev_sample
    return x.clamp(-1, 1)


def predict_x0_from_eps(x_t: torch.Tensor, t: torch.LongTensor, eps: torch.Tensor, scheduler: DDPMScheduler) -> torch.Tensor:
    """x0 = (x_t - sqrt(1 - a_bar_t) * eps) / sqrt(a_bar_t)"""
    a_bar = scheduler.alphas_cumprod.to(x_t.device)[t]   # (B,)
    a_bar = a_bar.view(-1, 1, 1, 1, 1)
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


# ===============================
# Eval saving
# ===============================
def save_eval_samples(
    samples: torch.Tensor,
    output_dir: str,
    epoch: int,
    threshold: float,
) -> None:
    """Save .binvox + a rendered PNG frame for each sample."""
    samples_np = samples.squeeze(1).detach().cpu().numpy()  # (B,D,H,W)
    out_samples_dir = os.path.join(output_dir, "samples")
    ensure_dir(out_samples_dir)
    for i, vol in enumerate(samples_np):
        vol_bin = (vol > threshold)
        binvox_path = os.path.join(out_samples_dir, f"epoch{epoch:04d}_sample{i:03d}.binvox")
        save_binvox(vol_bin.astype(np.bool_), binvox_path)
        img = vox_frame(vol, thr=threshold)
        Image.fromarray(img).save(os.path.join(out_samples_dir, f"epoch{epoch:04d}_sample{i:03d}.png"))


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

    model.eval()
    for b in range(num_batches):
        cur_bs = bs if (b + 1) * bs <= total else total - b * bs
        samples = sample_volumes(
            model=accelerator.unwrap_model(model),
            scheduler=scheduler,
            batch_size=cur_bs,
            voxel_size=config.voxel_size,
            num_inference_steps=config.num_inference_steps,
            device=device,
        )
        all_samples.append(samples)
    model.train()

    all_samples_t = torch.cat(all_samples, dim=0)
    save_eval_samples(all_samples_t, config.output_dir, epoch, threshold=config.binarize_threshold)

    # W&B: log a few eval frames
    previews = {}
    for i in range(min(4, all_samples_t.shape[0])):
        img = vox_frame(all_samples_t[i, 0].detach().cpu().numpy(), thr=config.binarize_threshold)
        previews[f"eval/sample_{i}"] = wandb.Image(img)
    if previews:
        wandb.log(previews, step=epoch)


# ===============================
# Train
# ===============================
def grad_global_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().data.norm(2).item() ** 2)
    return math.sqrt(total)


def train() -> None:
    # argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_dir", type=str, default="./ddpm-vox-03001627-32")
    parser.add_argument("--train_batch_size", type=int, default=32)
    args = parser.parse_args()
    config.train_batch_size = args.train_batch_size
    
    # load checkpoint 
    if os.path.exists(args.resume_dir):
        model_dir = os.path.join(args.resume_dir, "unet")
        sched_dir = os.path.join(args.resume_dir, "scheduler")
        model = UNet3DConditionModel.from_pretrained(model_dir)
        noise_scheduler = DDPMScheduler.from_pretrained(sched_dir)
    else:
        model = build_unet3d(voxel_size=config.voxel_size)
        noise_scheduler = DDPMScheduler(num_train_timesteps=config.num_train_timesteps, prediction_type="epsilon")
    
    set_seed(config.seed)
    ensure_dir(config.output_dir)
    

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with=None,  # use wandb directly
        project_dir=os.path.join(config.output_dir, "logs"),
    )

    if accelerator.is_main_process:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            mode=config.wandb_mode,
            config={**vars(config), "task": "unconditional_3d_ddpm"},
        )

    dataset = BinvoxDataset(
        root=config.dataset_root,
        class_id=config.class_id,
        voxel_size=config.voxel_size,
        augment=False,
    )
    train_dataloader = DataLoader(
        dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(
            total=len(train_dataloader),
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch}",
        )
        visuals_logged = 0

        for step, batch in enumerate(train_dataloader):
            clean_vols: torch.Tensor = batch["images"].to(accelerator.device)  # (B,1,D,H,W)
            noise = torch.randn_like(clean_vols)
            bs = clean_vols.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,),
                device=clean_vols.device, dtype=torch.int64
            )
            noisy = noise_scheduler.add_noise(clean_vols, noise, timesteps)

            with accelerator.accumulate(model):
                encoder_hidden_states = get_attention_hidden_states(model, bs, clean_vols.device)
                noise_pred = model(noisy, timesteps, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # metrics
            if accelerator.is_main_process:
                log_payload = {
                    "train/loss": float(loss.detach()),
                    "train/lr": lr_scheduler.get_last_lr()[0],
                    "train/step": global_step,
                    "train/epoch": epoch,
                }
                try:
                    log_payload["train/grad_global_norm"] = grad_global_norm(accelerator.unwrap_model(model))
                except Exception:
                    pass
                wandb.log(log_payload, step=global_step)

            # visuals + histograms
            should_log_visual = (
                accelerator.is_main_process
                and (global_step % config.log_interval_steps == 0)
                and (visuals_logged < config.max_train_visuals_per_epoch)
            )
            if should_log_visual:
                with torch.no_grad():
                    x0_pred = predict_x0_from_eps(noisy, timesteps, noise_pred, noise_scheduler)

                # histograms
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

                # single-frame voxel renders
                train_clean_img = vox_frame(clean_vols[0, 0].detach().cpu().numpy(), thr=config.binarize_threshold)
                train_noisy_img = vox_frame(noisy[0, 0].detach().cpu().numpy(), thr=config.binarize_threshold)
                pred_x0_img = vox_frame(x0_pred[0, 0].detach().cpu().numpy(), thr=config.binarize_threshold)

                # unconditional test generation (single sample)
                with torch.no_grad():
                    unet = accelerator.unwrap_model(model)
                    was_training = unet.training
                    unet.eval()
                    noise_scheduler.set_timesteps(config.num_inference_steps, device=accelerator.device)
                    gen = torch.Generator(device=accelerator.device).manual_seed(config.seed + global_step)
                    x = torch.randn((1, 1, config.voxel_size, config.voxel_size, config.voxel_size),
                                    device=accelerator.device, generator=gen)
                    for t in noise_scheduler.timesteps:
                        encoder_hidden_states = get_attention_hidden_states(unet, 1, accelerator.device)
                        eps = unet(x, t, encoder_hidden_states=encoder_hidden_states).sample
                        x = noise_scheduler.step(model_output=eps, timestep=t, sample=x).prev_sample
                    if was_training:
                        unet.train()
                test_pred_img = vox_frame(x[0, 0].clamp(-1, 1).detach().cpu().numpy(), thr=config.binarize_threshold)

                wandb.log(
                    {
                        "train3d/clean_frame": wandb.Image(train_clean_img),
                        "train3d/noisy_frame": wandb.Image(train_noisy_img),
                        "train3d/pred_x0_frame": wandb.Image(pred_x0_img),
                        "test/pred_frame": wandb.Image(test_pred_img),
                    },
                    step=global_step,
                )

                visuals_logged += 1

            progress_bar.update(1)
            global_step += 1

        # eval + save
        if accelerator.is_main_process:
            if ((epoch + 1) % config.save_samples_every == 0) or (epoch == config.num_epochs - 1):
                evaluate_and_log(accelerator, config, epoch, model, noise_scheduler)

            if ((epoch + 1) % config.save_model_every == 0) or (epoch == config.num_epochs - 1):
                unwrapped = accelerator.unwrap_model(model)
                model_dir = os.path.join(config.output_dir, "unet")
                sched_dir = os.path.join(config.output_dir, "scheduler")
                ensure_dir(model_dir); ensure_dir(sched_dir)
                unwrapped.save_pretrained(model_dir)
                noise_scheduler.save_pretrained(sched_dir)
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
