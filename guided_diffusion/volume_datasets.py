import math
import random
import os

import blobfile as bf
from mpi4py import MPI
import numpy as np
import torch as th
from torch.utils.data import DataLoader, Dataset


def load_volume_data(
    *,
    data_dir,
    batch_size,
    volume_size,
    class_cond=False,
    deterministic=False,
    random_flip=True,
    random_rotate=True,
):
    """
    For a 3D volumetric dataset, create a generator over (volumes, kwargs) pairs.

    Each volume is an NCDHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory containing .npy volume files.
    :param batch_size: the batch size of each returned pair.
    :param volume_size: the size to which volumes are resized/cropped.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_flip: if True, randomly flip the volumes for augmentation.
    :param random_rotate: if True, randomly rotate the volumes for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_volume_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = VolumeDataset(
        volume_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_flip=random_flip,
        random_rotate=random_rotate,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_volume_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["npy", "npz"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_volume_files_recursively(full_path))
    return results


class VolumeDataset(Dataset):
    def __init__(
        self,
        resolution,
        volume_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_flip=True,
        random_rotate=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_volumes = volume_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_flip = random_flip
        self.random_rotate = random_rotate

    def __len__(self):
        return len(self.local_volumes)

    def __getitem__(self, idx):
        path = self.local_volumes[idx]
        
        # Load volume data
        if path.endswith('.npy'):
            volume = np.load(path)
        elif path.endswith('.npz'):
            data = np.load(path)
            # Assume the volume data is stored under key 'volume' or first key
            if 'volume' in data.files:
                volume = data['volume']
            else:
                volume = data[data.files[0]]
        else:
            raise ValueError(f"Unsupported file format: {path}")
        
        # Ensure volume is the right shape and type
        if len(volume.shape) == 3:
            # Add channel dimension if missing (D, H, W) -> (C, D, H, W)
            volume = volume[None, ...]
        elif len(volume.shape) == 4 and volume.shape[0] > volume.shape[-1]:
            # If shape is (D, H, W, C), convert to (C, D, H, W)
            volume = np.transpose(volume, (3, 0, 1, 2))
        
        # Resize/crop to target resolution
        volume = self._resize_volume(volume, self.resolution)
        
        # Random augmentations
        if self.random_flip and random.random() < 0.5:
            # Random flip along one axis
            axis = random.choice([1, 2, 3])  # Don't flip channel axis
            volume = np.flip(volume, axis=axis)
        
        if self.random_rotate and random.random() < 0.3:
            # Random 90-degree rotation in one plane
            axes = random.choice([(1, 2), (1, 3), (2, 3)])
            k = random.choice([1, 2, 3])
            volume = np.rot90(volume, k=k, axes=axes)
        
        # Normalize to [-1, 1]
        volume = volume.astype(np.float32)
        if volume.max() > 1.0:
            volume = volume / 127.5 - 1
        else:
            volume = volume * 2 - 1
        
        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        
        return volume, out_dict

    def _resize_volume(self, volume, target_size):
        """Resize volume to target size using interpolation or cropping."""
        c, d, h, w = volume.shape
        
        if d == h == w == target_size:
            return volume
        
        # Convert to torch tensor for easy resizing
        volume_tensor = th.from_numpy(volume).unsqueeze(0)  # Add batch dim
        
        # Resize using trilinear interpolation
        volume_tensor = th.nn.functional.interpolate(
            volume_tensor, size=(target_size, target_size, target_size), 
            mode='trilinear', align_corners=False
        )
        
        return volume_tensor.squeeze(0).numpy()


def random_crop_volume(volume, size):
    """Randomly crop a volume to the specified size."""
    c, d, h, w = volume.shape
    
    if d < size or h < size or w < size:
        # Pad if volume is smaller than target size
        pad_d = max(0, size - d)
        pad_h = max(0, size - h) 
        pad_w = max(0, size - w)
        volume = np.pad(volume, ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
        d, h, w = volume.shape[1:]
    
    # Random crop
    start_d = random.randint(0, d - size)
    start_h = random.randint(0, h - size)
    start_w = random.randint(0, w - size)
    
    return volume[:, start_d:start_d + size, start_h:start_h + size, start_w:start_w + size]


def center_crop_volume(volume, size):
    """Center crop a volume to the specified size."""
    c, d, h, w = volume.shape
    
    if d < size or h < size or w < size:
        # Pad if volume is smaller than target size
        pad_d = max(0, size - d)
        pad_h = max(0, size - h)
        pad_w = max(0, size - w)
        volume = np.pad(volume, ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
        d, h, w = volume.shape[1:]
    
    # Center crop
    start_d = (d - size) // 2
    start_h = (h - size) // 2
    start_w = (w - size) // 2
    
    return volume[:, start_d:start_d + size, start_h:start_h + size, start_w:start_w + size] 