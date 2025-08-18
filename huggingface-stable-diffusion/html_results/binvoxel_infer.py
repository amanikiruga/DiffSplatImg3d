# binvoxel_infer.py
#!/usr/bin/env python3
# Unconditional sampling for a trained 3D DDPM (UNet3DConditionModel + DDPMScheduler).
# Loads from <ckpt_dir>/unet and <ckpt_dir>/scheduler, generates N samples in batches,
# saves PNG frames (and optional .binvox) plus a manifest.json for the HTML viewer.

import sys 
sys.path.append("/om/user/akiruga/diffsplatimg3d/huggingface-stable-diffusion")
import os
import math
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from diffusers import DDPMScheduler
from diffusers.models import UNet3DConditionModel

import binvox_rw
from tqdm import tqdm


def get_attention_hidden_states(model: UNet3DConditionModel, batch_size: int, device: torch.device) -> torch.Tensor:
    """Create dummy zero conditioning matching training behavior.
    If the model has no cross-attention, we still pass a (B,1,1) tensor.
    """
    cond_dim = getattr(model, "config", None)
    cond_dim = getattr(cond_dim, "cross_attention_dim", None) if cond_dim is not None else None
    cond_dim = cond_dim if cond_dim is not None else 1
    return torch.zeros((batch_size, 1, cond_dim), device=device, dtype=torch.float32)


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_binvox(volume_bool: np.ndarray, out_path: Path) -> None:
    assert volume_bool.dtype == np.bool_, "Expected boolean array for binvox."
    dims = list(volume_bool.shape)
    vox = binvox_rw.Voxels(volume_bool, dims=dims, translate=[0.0, 0.0, 0.0], scale=1.0, axis_order="xyz")
    with open(out_path, "wb") as f:
        vox.write(f)


def vox_frame(vol_dhw: np.ndarray, thr: float = 0.0) -> np.ndarray:
    # first rot 90 degrees 
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


@torch.no_grad()
def reverse_diffusion(unet: UNet3DConditionModel,
                      scheduler: DDPMScheduler,
                      batch_size: int,
                      voxel_size: int,
                      num_inference_steps: int,
                      device: torch.device) -> torch.Tensor:
    scheduler.set_timesteps(num_inference_steps, device=device)
    x = torch.randn((batch_size, 1, voxel_size, voxel_size, voxel_size), device=device)
    # Match training: always provide dummy zero conditioning, even if cross-attention is disabled
    enc_hid = get_attention_hidden_states(unet, batch_size, device)

    for t in tqdm(scheduler.timesteps, desc="Reverse diffusion"):
        eps = unet(x, t, encoder_hidden_states=enc_hid).sample
        x = scheduler.step(model_output=eps, timestep=t, sample=x).prev_sample

    return x.clamp(-1, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Training output dir containing 'unet' and 'scheduler' subdirs.")
    parser.add_argument("--num_samples", "-n", type=int, default=64)
    parser.add_argument("--batch_size", "-b", type=int, default=32)
    parser.add_argument("--num_inference_steps", type=int, default=250)
    parser.add_argument("--threshold", type=float, default=0.0, help="Binarization/render threshold in [-1,1].")
    parser.add_argument("--out_dir", type=str, default=None, help="Where to save outputs; defaults to <ckpt_dir>/inference/<timestamp>")
    parser.add_argument("--save_binvox", action="store_true", help="Also save .binvox alongside PNGs.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)

    ckpt_dir = Path(args.ckpt_dir)
    unet_dir = ckpt_dir / "unet"
    sched_dir = ckpt_dir / "scheduler"

    assert unet_dir.is_dir(), f"Missing {unet_dir}"
    assert sched_dir.is_dir(), f"Missing {sched_dir}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    unet = UNet3DConditionModel.from_pretrained(unet_dir).to(device)
    print(f"Loaded UNet from {unet_dir}")
    unet.eval()
    scheduler = DDPMScheduler.from_pretrained(sched_dir)
    print(f"Loaded scheduler from {sched_dir}")

    voxel_size = unet.config.sample_size
    assert isinstance(voxel_size, int), f"Unexpected sample_size: {voxel_size}"
    print(f"Voxel size: {voxel_size}")

    out_root = Path(args.out_dir) if args.out_dir else (ckpt_dir / "inference" / datetime.now().strftime("%Y%m%d-%H%M%S"))
    img_dir = out_root / "images"
    vox_dir = out_root / "binvox"
    ensure_dir(img_dir)
    if args.save_binvox:
        ensure_dir(vox_dir)
    print(f"Output directory: {out_root}")
    remaining = args.num_samples
    bs = max(1, args.batch_size)
    sample_idx = 0
    manifest = {
        "num_samples": args.num_samples,
        "voxel_size": voxel_size,
        "num_inference_steps": args.num_inference_steps,
        "threshold": args.threshold,
        "seed": args.seed,
        "images": []  # list of {"png": "images/...", "binvox": "binvox/..."} (binvox optional)
    }

    loops = math.ceil(remaining / bs)
    for _ in tqdm(range(loops), desc="Generating samples"):
        cur_bs = min(bs, remaining)
        vols = reverse_diffusion(unet, scheduler, cur_bs, voxel_size, args.num_inference_steps, device)  # (B,1,D,H,W)
        vols_np = vols.squeeze(1).detach().cpu().numpy()  # (B,D,H,W)

        for i in tqdm(range(cur_bs), desc="Saving samples"):
            png_name = f"sample_{sample_idx:05d}.png"
            png_path = img_dir / png_name
            img = vox_frame(vols_np[i], thr=args.threshold)
            Image.fromarray(img).save(png_path)

            entry = {"png": f"images/{png_name}"}

            if args.save_binvox:
                bv_name = f"sample_{sample_idx:05d}.binvox"
                bv_path = vox_dir / bv_name
                save_binvox((vols_np[i] > args.threshold).astype(np.bool_), bv_path)
                entry["binvox"] = f"binvox/{bv_name}"

            manifest["images"].append(entry)
            sample_idx += 1

        remaining -= cur_bs
        print(f"[infer] generated {sample_idx}/{args.num_samples}")

    with open(out_root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[infer] done. outputs in: {out_root.resolve()}")


if __name__ == "__main__":
    main()
