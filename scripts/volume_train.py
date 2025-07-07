"""
Train a 3D diffusion model on volumetric data.

Training details:
- Batch size of 8
- Adam optimizer with initial learning rate of 10^-4
- Linear beta scheduling from 0.0015 to 0.05 at 1000 timesteps
- Custom loss weighting with ωt = ᾱ²t
- Train for 3.0M iterations with decaying LR from 10^-4 to 10^-6
- Voxel grid resolution of 32
"""

import argparse
import numpy as np
import torch as th

from guided_diffusion import dist_util, logger
from guided_diffusion.volume_datasets import load_volume_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util_3d import (
    model_and_diffusion_defaults_3d,
    create_model_and_diffusion_3d,
    args_to_dict,
    add_dict_to_argparser,
    create_model_3d,
    str2bool,
)
from guided_diffusion.train_util import TrainLoop
from guided_diffusion import gaussian_diffusion as gd


def create_custom_beta_schedule(steps=1000, beta_start=0.0015, beta_end=0.05):
    """
    Create custom linear beta schedule from beta_start to beta_end.
    """
    return np.linspace(beta_start, beta_end, steps, dtype=np.float64)


def create_custom_diffusion_3d(
    *,
    steps=1000,
    learn_sigma=False,
    beta_start=0.0015,
    beta_end=0.05,
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    """
    Create a custom Gaussian diffusion process for 3D data with specified beta schedule.
    """
    from guided_diffusion.respace import SpacedDiffusion, space_timesteps
    
    # Use custom beta schedule
    betas = create_custom_beta_schedule(steps, beta_start, beta_end)
    
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not learn_sigma
                else gd.ModelVarType.LEARNED_RANGE
            )
            if not rescale_learned_sigmas
            else gd.ModelVarType.LEARNED
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


class Custom3DTrainLoop(TrainLoop):
    """
    Custom training loop for 3D diffusion with specialized loss weighting.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Pre-compute alpha_bar values for loss weighting
        self.alpha_bar = self.diffusion.alphas_cumprod
    
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
                
                terms = {}
                model_output = self.ddp_model(x_t, t, **micro_cond)
                
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
                alpha_bar_t = self.alpha_bar[t].to(micro.device)
                custom_weights = alpha_bar_t ** 2
                
                terms["mse"] = mse_loss
                terms["loss"] = mse_loss * custom_weights
                return terms

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, self.diffusion.__class__):
                pass  # No special handling needed for uniform sampler
            
            loss = (losses["loss"] * weights).mean()
            
            # Log losses
            logger.logkv_mean("train_loss", loss.item())
            logger.logkv_mean("train_mse", (losses["mse"] * weights).mean().item())
            
            self.mp_trainer.backward(loss)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating 3D model and custom diffusion...")
    
    # Create model using 3D utilities
    model = create_model_3d(
        volume_size=args.volume_size,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        num_classes=args.num_classes if args.class_cond else None,
        dropout=args.dropout,
        use_checkpoint=args.use_checkpoint,
        use_fp16=args.use_fp16,
        use_scale_shift_norm=args.use_scale_shift_norm,
        resblock_updown=args.resblock_updown,
        use_new_attention_order=args.use_new_attention_order,
    )
    
    # Create custom diffusion process
    diffusion = create_custom_diffusion_3d(
        steps=args.diffusion_steps,
        learn_sigma=args.learn_sigma,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        use_kl=args.use_kl,
        predict_xstart=args.predict_xstart,
        rescale_timesteps=args.rescale_timesteps,
        rescale_learned_sigmas=args.rescale_learned_sigmas,
        timestep_respacing=args.timestep_respacing,
    )
    
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating 3D data loader...")
    data = load_volume_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        volume_size=args.volume_size,
        class_cond=args.class_cond,
    )

    logger.log("training 3D diffusion model...")
    Custom3DTrainLoop(
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
    ).run_loop()


def create_argparser():
    defaults = dict(
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
        # 3D specific parameters
        volume_size=32,  # Voxel grid resolution
        in_channels=3,
        out_channels=3,
        class_cond=False,
        # Custom diffusion parameters
        beta_start=0.0015,
        beta_end=0.05,
        diffusion_steps=1000,
        learn_sigma=False,
        noise_schedule="linear",  # We override this with custom schedule
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        # Model architecture parameters
        num_classes=None,
        dropout=0.0,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_new_attention_order=False,
    )
    
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    
    # Add custom arguments
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
        "--volume_size", 
        type=int, 
        default=32,
        help="3D volume resolution (cubic)"
    )
    
    return parser


# Note: add_dict_to_argparser and str2bool are imported from script_util_3d


if __name__ == "__main__":
    main() 