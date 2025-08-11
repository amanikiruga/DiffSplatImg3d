"""
Helpers for distributed training — compatible with both torchrun and MPI.
"""

import io
import os
import socket

import blobfile as bf
import torch as th
import torch.distributed as dist


try:
    from mpi4py import MPI  # optional, used only if torchrun env‑vars are absent
except ImportError:
    MPI = None

# Adjust to your node’s GPU count.
GPUS_PER_NODE = 8


# ────────────────────────── initialization ────────────────────────── #

def _using_torchrun_env() -> bool:
    """Detect torchrun / torch.distributed.run environment variables."""
    return all(k in os.environ for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))


def _find_free_port() -> int:
    """Pick an unused TCP port on the current host (rank 0 only)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def setup_dist() -> None:
    """
    Initialize torch.distributed for either torchrun or MPI.
    Safe to call more than once.
    """
    if dist.is_initialized():
        return

    backend = "nccl" if th.cuda.is_available() else "gloo"

    # ── Preferred path: torchrun ─────────────────────────────────────
    if _using_torchrun_env():
        dist.init_process_group(backend=backend, init_method="env://")
        return

    # ── Fallback: set env‑vars using mpi4py ──────────────────────────
    if MPI is None:
        raise RuntimeError(
            "Distributed vars not set and mpi4py unavailable; "
            "launch with torchrun or install mpi4py."
        )

    comm = MPI.COMM_WORLD

    # One GPU per rank (round‑robin if ranks > GPUs).
    if th.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(comm.rank % GPUS_PER_NODE)

    # Populate master address/port.
    hostname = socket.gethostbyname(socket.getfqdn())
    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    os.environ["MASTER_PORT"] = str(comm.bcast(_find_free_port(), root=0))

    # Rank and world size.
    os.environ["RANK"] = str(comm.rank)
    os.environ["WORLD_SIZE"] = str(comm.size)

    dist.init_process_group(backend=backend, init_method="env://")


# ───────────────────────────── utilities ──────────────────────────── #

def dev() -> th.device:
    """Return the correct torch.device for this process."""
    if not th.cuda.is_available():
        return th.device("cpu")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return th.device(f"cuda:{local_rank}")


def load_state_dict(path: str, **torch_load_kwargs):
    """
    Rank‑0 blobfile load with broadcast.
    • If MPI is active → broadcast bytes manually (avoids duplicate I/O).
    • Otherwise → fall back to torch.load on every rank (torchrun path).
    """
    return th.load(path, **torch_load_kwargs)

    if MPI is None or not MPI.Is_initialized():
        return th.load(path, **torch_load_kwargs)

    comm = MPI.COMM_WORLD
    chunk_sz = 1 << 30  # 1 GiB

    if comm.rank == 0:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()
        num_chunks = (len(data) + chunk_sz - 1) // chunk_sz
    else:
        data = None
        num_chunks = None

    num_chunks = comm.bcast(num_chunks, root=0)
    buf = bytearray()

    for idx in range(num_chunks):
        if comm.rank == 0:
            chunk = memoryview(data)[idx * chunk_sz : (idx + 1) * chunk_sz]
        else:
            chunk = None
        chunk = comm.bcast(chunk, root=0)
        buf.extend(chunk)

    return th.load(io.BytesIO(buf), **torch_load_kwargs)


def sync_params(params) -> None:
    """Broadcast parameters from rank 0 to all ranks."""
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)
