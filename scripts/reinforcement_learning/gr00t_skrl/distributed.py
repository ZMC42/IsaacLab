# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Small rank coordination messages; parameter and gradient synchronization stay in skrl."""

from __future__ import annotations

import hashlib
import random
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from .protocol import RunConfig


def gather_rank_objects(value: object) -> list:
    """Exchange small CPU objects using the model process's native skrl group.

    Args:
        value: Small picklable counters, hashes or RNG state; never model tensors.

    Returns:
        One object per global rank, in rank order.
    """
    if not dist.is_initialized():
        return [value]
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


def require_rank_agreement(label: str, value: object) -> list:
    """Fail every rank if a small contract or identity differs.

    Args:
        label: Contract name used in the error.
        value: Comparable, picklable local value.

    Returns:
        Agreed values in rank order.
    """
    values = gather_rank_objects(value)
    if any(item != values[0] for item in values[1:]):
        raise ValueError(f"Distributed {label} mismatch across ranks: {values}")
    return values


def rank_seed(base_seed: int, rank: int, updates: int = 0) -> int:
    """Derive independent streams while preserving the original single-rank seed.

    Args:
        base_seed: Run's original seed.
        rank: New global rank.
        updates: Completed updates, including those before a topology change.

    Returns:
        Seed accepted by Python, NumPy and Torch.
    """
    if rank == 0 and updates == 0:
        return base_seed
    digest = hashlib.sha256(f"{base_seed}:{updates}:{rank}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def seed_rng(seed: int) -> None:
    """Seed the model's Python, NumPy and Torch random streams."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def check_rank_config(cfg: RunConfig) -> list[dict]:
    """Check local rollout contracts before loading assets and record the physical rank mapping.

    Args:
        cfg: Launcher-written rank configuration.

    Returns:
        Current rank topology used to decide whether checkpoint RNG can be reused.
    """
    if cfg.world_size > 1 and (
        not dist.is_initialized() or (dist.get_rank(), dist.get_world_size()) != (cfg.rank, cfg.world_size)
    ):
        raise RuntimeError("Model process did not join the expected skrl communication group")
    contract = asdict(cfg)
    for name in ("rank", "local_rank", "node_rank", "node_id", "gpu_id", "run_dir", "socket_path", "viser"):
        contract.pop(name)
    require_rank_agreement("run configuration", contract)
    return gather_rank_objects(
        {"node_id": cfg.node_id, "node_rank": cfg.node_rank, "local_rank": cfg.local_rank, "gpu_id": cfg.gpu_id}
    )


def runner_digest() -> str:
    """Fingerprint the runner source on each node without sending source over NCCL."""
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


@contextmanager
def measure_phase(seconds: dict[str, float], name: str, device: torch.device) -> Iterator[None]:
    """Accumulate wall time [s] including completion of CUDA work for one phase.

    Args:
        seconds: Accumulated phase times [s].
        name: Phase name.
        device: Model device to synchronize.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.monotonic()
    try:
        yield
    finally:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds[name] = seconds.get(name, 0.0) + time.monotonic() - start
