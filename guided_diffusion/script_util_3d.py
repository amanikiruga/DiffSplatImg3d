"""
Script utilities for creating 3D U-Net models.
"""

import argparse

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .unet_3d import UNet3DModel


def model_and_diffusion_defaults_3d():
    """
    Defaults for 3D U-Net model and diffusion process.
    """
    res = dict(
        volume_size=64,
        in_channels=3,
        out_channels=3,
        num_classes=None,
        dropout=0.0,
        use_checkpoint=False,
        use_fp16=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_new_attention_order=False,
        learn_sigma=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )
    return res


def create_model_3d(
    volume_size,
    in_channels,
    out_channels,
    num_classes,
    dropout,
    use_checkpoint,
    use_fp16,
    use_scale_shift_norm,
    resblock_updown,
    use_new_attention_order,
    **kwargs
):
    """
    Create a 3D U-Net model with the specified architecture:
    - 4 scaling blocks with 2 ResNet blocks per scale
    - Linear increase from 64 to 256 channels
    - Skip attention blocks at scaling factors 2, 4, and 8 with 32 channels per head
    """
    return UNet3DModel(
        volume_size=volume_size,
        in_channels=in_channels,
        out_channels=out_channels,
        dropout=dropout,
        num_classes=num_classes,
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )


def create_gaussian_diffusion_3d(
    *,
    steps=1000,
    learn_sigma=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    """
    Create a Gaussian diffusion process for 3D data.
    """
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
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


def create_model_and_diffusion_3d(
    volume_size,
    in_channels,
    out_channels,
    num_classes,
    dropout,
    use_checkpoint,
    use_fp16,
    use_scale_shift_norm,
    resblock_updown,
    use_new_attention_order,
    learn_sigma,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
):
    """
    Create both a 3D U-Net model and diffusion process.
    """
    model = create_model_3d(
        volume_size=volume_size,
        in_channels=in_channels,
        out_channels=out_channels,
        num_classes=num_classes,
        dropout=dropout,
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )
    diffusion = create_gaussian_diffusion_3d(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion


def add_dict_to_argparser(parser, default_dict):
    """
    Add a dictionary of default values to an argparser.
    """
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    """
    Extract a subset of arguments as a dictionary.
    """
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    Convert string to boolean for argparse.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.") 