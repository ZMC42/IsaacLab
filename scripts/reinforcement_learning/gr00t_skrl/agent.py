# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lifecycle extensions only; PPO updates, GAE, losses and serialization remain native skrl."""

from __future__ import annotations

import hashlib
import json
import os
import random
import resource
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from skrl.agents.torch.ppo import PPO

from .distributed import gather_rank_objects, measure_phase, rank_seed, require_rank_agreement, seed_rng
from .policy import module_digest
from .protocol import RunConfig

if TYPE_CHECKING:
    from .policy import FrozenEncoder


def compatible_metadata(saved: dict, current: dict) -> bool:
    """Interpret legacy single-environment checkpoints as training batches of one."""
    return {"training_batch_size": 1, **saved} == {"training_batch_size": 1, **current}


class RunState:
    """Native checkpoint module for cumulative counters, compatibility metadata and RNG."""

    def __init__(self, metadata: dict, *, cfg: RunConfig | None = None, topology: list[dict] | None = None):
        self.metadata = metadata
        self.environment_steps = 0
        self.transitions = 0
        self.updates = 0
        self.optimizer_steps = 0
        self.pending_rng: dict | None = None
        self.pending_seed: int | None = None
        self.seed = cfg.seed if cfg is not None else 42
        self.rank = cfg.rank if cfg is not None else 0
        self.world_size = cfg.world_size if cfg is not None else 1
        self.num_envs = cfg.num_envs if cfg is not None else 1
        self.topology = topology or []
        self.rank_rng: list[dict] | None = None

    def capture_rng(self) -> dict:
        """Capture Python, NumPy, CPU Torch and only this rank's current GPU RNG."""
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": [torch.cuda.get_rng_state()] if torch.cuda.is_available() else [],
        }

    def counters(self) -> dict[str, int]:
        """Return shared cumulative counters for update-boundary agreement."""
        return {
            "environment_steps": self.environment_steps,
            "transitions": self.transitions,
            "updates": self.updates,
            "optimizer_steps": self.optimizer_steps,
        }

    def state_dict(self) -> dict:
        """Capture counters and RNG at a complete native update boundary."""
        rng = self.rank_rng or [self.capture_rng()]
        return {
            "metadata": self.metadata,
            **self.counters(),
            "rng": rng[0],
            "distributed": {
                "version": 1,
                "seed": self.seed,
                "world_size": self.world_size,
                "num_envs": self.num_envs,
                "topology": self.topology,
                "rank_rng": rng,
            },
        }

    def load_state_dict(self, state: dict) -> None:
        """Reject incompatible policy contracts and defer RNG restoration until initialization ends."""
        if not compatible_metadata(state.get("metadata", {}), self.metadata):
            raise ValueError("Incompatible GR00T skrl checkpoint metadata")
        self.environment_steps = state["environment_steps"]
        self.transitions = state.get("transitions", self.environment_steps)
        self.updates = state["updates"]
        self.optimizer_steps = state["optimizer_steps"]
        self.pending_rng = None
        self.pending_seed = None
        distribution = state.get("distributed")
        if distribution is None:
            if self.world_size == 1 and self.num_envs == 1:
                self.pending_rng = state["rng"]
            else:
                self.pending_seed = rank_seed(self.seed, self.rank, self.updates)
        else:
            self.seed = distribution["seed"]
            same_topology = (
                distribution["world_size"] == self.world_size
                and distribution["num_envs"] == self.num_envs
                and distribution["topology"] == self.topology
            )
            if same_topology:
                self.pending_rng = distribution["rank_rng"][self.rank]
            else:
                self.pending_seed = rank_seed(self.seed, self.rank, self.updates)

    def restore_rng(self) -> None:
        """Restore after native load and all necessary model/trainer initialization."""
        if self.pending_seed is not None:
            seed_rng(self.pending_seed)
            self.pending_seed = None
        elif self.pending_rng is not None:
            random.setstate(self.pending_rng["python"])
            np.random.set_state(self.pending_rng["numpy"])
            torch.set_rng_state(self.pending_rng["torch"].cpu())
            if self.pending_rng["cuda"]:
                torch.cuda.set_rng_state(self.pending_rng["cuda"][0].cpu())
            self.pending_rng = None


def tensor_state_digest(state: dict) -> str:
    """Fingerprint native state tensors and scalar metadata for independent identity checks."""
    digest = hashlib.sha256()

    def update(value):
        digest.update(type(value).__name__.encode())
        if isinstance(value, dict):
            for name, item in value.items():
                update(name)
                update(item)
        elif isinstance(value, (tuple, list)):
            digest.update(str(len(value)).encode())
            for item in value:
                update(item)
        elif isinstance(value, torch.Tensor):
            digest.update(f"{value.dtype}:{tuple(value.shape)}".encode())
            digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, np.ndarray):
            digest.update(f"{value.dtype}:{value.shape}".encode())
            digest.update(value.tobytes())
        else:
            digest.update(repr(value).encode())

    update(state)
    return digest.hexdigest()


def validate_checkpoint(path: str, metadata: dict) -> dict:
    """Read trusted local checkpoint on CPU and reject old or incompatible formats before native load."""
    modules = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(modules, dict) or set(modules) != {"policy", "value", "optimizer", "run_state"}:
        raise ValueError("Expected a GR00T native skrl checkpoint; old handwritten PPO is unsupported")
    state = modules["run_state"]
    distribution = state.get("distributed")
    if distribution is not None and (
        distribution.get("version") != 1
        or distribution.get("world_size", 0) < 1
        or len(distribution.get("rank_rng", [])) != distribution["world_size"]
    ):
        raise ValueError("Invalid distributed checkpoint state")
    if not compatible_metadata(state.get("metadata", {}), metadata):
        raise ValueError("Incompatible GR00T skrl checkpoint metadata")
    steps = [int(item["step"].item()) for item in modules["optimizer"]["state"].values()]
    if not steps or min(steps) != state["optimizer_steps"] or max(steps) != state["optimizer_steps"]:
        raise ValueError("Checkpoint optimizer steps disagree with cumulative run state")
    result = {
        "optimizer_steps": state["optimizer_steps"],
        "environment_steps": state["environment_steps"],
        "transitions": state.get("transitions", state["environment_steps"]),
        "updates": state["updates"],
        "policy_state_sha256": tensor_state_digest(modules["policy"]),
        "value_state_sha256": tensor_state_digest(modules["value"]),
        "optimizer_state_sha256": tensor_state_digest(modules["optimizer"]["state"]),
        "optimizer_groups_sha256": tensor_state_digest({"groups": modules["optimizer"]["param_groups"]}),
        "run_state_sha256": tensor_state_digest(state),
    }
    del modules
    return result


class Gr00tPPO(PPO):
    """Thin native agent hooks for terminal states, backbone residency and boundary checkpoints."""

    def __init__(
        self,
        *,
        encoder: FrozenEncoder | None,
        run_state: RunState,
        diagnostic_batch_size: int = 1,
        checkpoint_interval: int = 1,
        verify_distributed_state: bool = False,
        warmup_updates: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.run_state = run_state
        self.diagnostic_batch_size = diagnostic_batch_size
        self.checkpoint_modules["run_state"] = run_state
        self.diagnostics: list[dict] = []
        self.update_checkpoint_interval = checkpoint_interval
        self.verify_distributed_state = verify_distributed_state
        self.last_saved_update = run_state.updates
        self.phase_seconds: dict[str, float] = {}
        self.warmup_updates = warmup_updates
        self.measurement_start: float | None = None
        self.measurement_counters: dict[str, int] | None = None
        self._invocation_start_updates = run_state.updates
        self._resume_offset = run_state.environment_steps
        self._gradient_norm = 0.0
        self._finite_gradients = True
        self.latest_losses: dict[str, float] = {}
        self.optimizer.param_groups[0]["foreach"] = False
        self.optimizer.register_step_pre_hook(self._before_optimizer_step)
        self.optimizer.register_step_post_hook(self._after_optimizer_step)

    def act(self, *args, **kwargs) -> tuple:
        """Time native action sampling without changing the saved generation chain."""
        with measure_phase(self.phase_seconds, "sampling", self.device):
            return super().act(*args, **kwargs)

    def record_transition(
        self, *, next_states: torch.Tensor, terminated: torch.Tensor, truncated: torch.Tensor, infos: dict, **kwargs
    ) -> None:
        """Let native PPO bootstrap only timeouts using reset-before-terminal critic state."""
        timeout = truncated & ~terminated
        if self.training and timeout.any():
            final_state = infos.get("final_state")
            if final_state is None or final_state.shape != next_states.shape:
                raise ValueError("Timeout transition requires reset-before-terminal final_state")
            next_states = torch.where(timeout, final_state, next_states)
        super().record_transition(
            next_states=next_states, terminated=terminated, truncated=timeout, infos=infos, **kwargs
        )
        self.run_state.environment_steps += 1
        self.run_state.transitions += next_states.shape[0] * self.run_state.world_size

    def track_data(self, tag: str, value: float) -> None:
        """Retain native loss diagnostics and fail immediately on non-finite losses."""
        if tag.startswith("Loss /"):
            if not np.isfinite(value):
                raise FloatingPointError(f"Non-finite native {tag}: {value}")
            self.latest_losses[tag] = value
        super().track_data(tag, value)

    def write_tracking_data(self, *, timestep: int, timesteps: int) -> None:
        """Use cumulative environment steps for native TensorBoard after a fresh trainer resumes."""
        self.track_data("Stats / Synchronous steps", self.run_state.environment_steps)
        self.track_data("Stats / Transitions", self.run_state.transitions)
        super().write_tracking_data(timestep=timestep + self._resume_offset, timesteps=timesteps + self._resume_offset)

    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        """Wrap native post_interaction without replacing its update or optimizer logic."""
        updating = (
            self.training and (self._rollout + 1) % self.cfg.rollouts == 0 and timestep >= self.cfg.learning_starts
        )
        if not updating:
            super().post_interaction(timestep=timestep, timesteps=timesteps)
            return
        with measure_phase(self.phase_seconds, "diagnostics", self.device):
            before = module_digest(self.policy)
        if self.encoder is not None:
            with measure_phase(self.phase_seconds, "backbone_offload", self.device):
                self.encoder.offload()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        try:
            with measure_phase(self.phase_seconds, "ppo_update", self.device):
                super().post_interaction(timestep=timestep, timesteps=timesteps)
            self.run_state.updates += 1
            with measure_phase(self.phase_seconds, "diagnostics", self.device):
                diagnostics = self._diagnose_update(before)
            self.diagnostics.append(diagnostics)
            if self.run_state.updates % self.update_checkpoint_interval == 0:
                self.save_boundary()
            diagnostics["save_peak_allocated_gib"] = (
                torch.cuda.max_memory_allocated(self.device) / 2**30 if self.device.type == "cuda" else 0
            )
            print(json.dumps({"rank": self.run_state.rank, **diagnostics}), flush=True)
        finally:
            if self.encoder is not None:
                with measure_phase(self.phase_seconds, "backbone_restore", self.device):
                    self.encoder.restore()
        if self.warmup_updates and self.run_state.updates - self._invocation_start_updates == self.warmup_updates:
            require_rank_agreement("warmup boundary", self.run_state.counters())
            self.phase_seconds.clear()
            self.measurement_counters = self.run_state.counters()
            self.measurement_start = time.monotonic()

    def _diagnose_update(self, before: str) -> dict:
        """Check the native update and evaluate saved chains with the updated policy."""
        after = module_digest(self.policy)
        optimizer_steps = [int(state["step"].item()) for state in self.optimizer.state.values() if "step" in state]
        if not optimizer_steps or max(optimizer_steps) != self.run_state.optimizer_steps:
            raise RuntimeError("Native optimizer and cumulative step counters disagree")
        if not self._finite_gradients or before == after:
            raise FloatingPointError("Non-finite gradients or unchanged original head")
        log_ratios = []
        with torch.no_grad():
            observations = self.memory.get_tensor_by_name("observations").flatten(0, 1)
            actions = self.memory.get_tensor_by_name("actions").flatten(0, 1)
            previous = self.memory.get_tensor_by_name("log_prob").flatten(0, 1)
            for start in range(0, observations.shape[0], self.diagnostic_batch_size):
                batch = slice(start, start + self.diagnostic_batch_size)
                _, output = self.policy.act({"observations": observations[batch], "taken_actions": actions[batch]})
                log_ratios.append(output["log_prob"] - previous[batch])
        log_ratios = torch.cat(log_ratios)
        ratios = log_ratios.exp()
        if not torch.isfinite(ratios).all():
            raise FloatingPointError("Non-finite post-update PPO ratio")
        diagnostics = {
            "native_losses": self.latest_losses.copy(),
            "ratio_min": float(ratios.min()),
            "ratio_max": float(ratios.max()),
            "approximate_kl": float((ratios - 1 - log_ratios).mean()),
            "clip_fraction": float(((ratios - 1).abs() > self.cfg.ratio_clip).float().mean()),
            "recomputed_after_optimizer_step": len(getattr(self.policy, "recomputed_versions", set())) > 1,
            "environment_steps": self.run_state.environment_steps,
            "transitions": self.run_state.transitions,
            "updates": self.run_state.updates,
            "optimizer_steps": self.run_state.optimizer_steps,
            "head_changed": before != after,
            "gradient_norm_after_clip": self._gradient_norm,
            "finite_gradients": self._finite_gradients,
            "head_sha256": after,
            "update_peak_allocated_gib": torch.cuda.max_memory_allocated(self.device) / 2**30
            if self.device.type == "cuda"
            else 0,
        }
        return diagnostics

    def verify_state(self) -> dict:
        """Compare policy, critic, Adam and shared counters using only small fingerprints."""
        state = {
            **self.run_state.counters(),
            "policy": module_digest(self.policy),
            "value": module_digest(self.value),
            "optimizer": tensor_state_digest(self.optimizer.state_dict()),
        }
        require_rank_agreement("policy/critic/Adam state", state)
        return state

    def save_boundary(self) -> None:
        """Publish one complete native checkpoint atomically, or fail every rank.

        RNG and counters are collected before rank 0 writes; every rank waits for the write result.
        Repeated calls at the same update boundary do not create a second checkpoint.
        """
        if self.last_saved_update == self.run_state.updates:
            return
        with measure_phase(self.phase_seconds, "checkpoint", self.device):
            require_rank_agreement("checkpoint counters", self.run_state.counters())
            if self.verify_distributed_state:
                self.verify_state()
            self.run_state.rank_rng = gather_rank_objects(self.run_state.capture_rng())
            error = None
            temporary_path = None
            try:
                if self.run_state.rank == 0:
                    try:
                        checkpoint_dir = Path(self.experiment_dir) / "checkpoints"
                        checkpoint_dir.mkdir(parents=True, exist_ok=True)
                        destination = checkpoint_dir / f"agent_{self.run_state.environment_steps}.pt"
                        descriptor, temporary_path = tempfile.mkstemp(
                            prefix=".agent-", suffix=".tmp", dir=checkpoint_dir
                        )
                        os.close(descriptor)
                        self.save(temporary_path)
                        with open(temporary_path, "rb") as stream:
                            os.fsync(stream.fileno())
                        os.replace(temporary_path, destination)
                        descriptor = os.open(checkpoint_dir, os.O_RDONLY | os.O_DIRECTORY)
                        try:
                            os.fsync(descriptor)
                        finally:
                            os.close(descriptor)
                    except Exception as exception:
                        error = f"{type(exception).__name__}: {exception}"
                errors = gather_rank_objects(error)
                if any(errors):
                    raise RuntimeError(f"Checkpoint save failed: {errors}")
                self.last_saved_update = self.run_state.updates
            finally:
                self.run_state.rank_rng = None
                if temporary_path is not None:
                    with suppress(FileNotFoundError):
                        os.unlink(temporary_path)

    def resume(self, path: str, *, summary: dict | None = None) -> dict:
        """Validate on CPU, then use native load on CPU to avoid duplicating GPU checkpoint tensors."""
        summary = summary or validate_checkpoint(path, self.run_state.metadata)
        require_rank_agreement("checkpoint identity", summary)
        if self.encoder is not None:
            self.encoder.offload()
        device = self.device
        self.policy.to("cpu")
        self.value.to("cpu")
        self.device = torch.device("cpu")
        try:
            super().load(path)
        finally:
            self.device = device
        self.policy.to(device)
        self.value.to(device)
        for state in self.optimizer.state.values():
            for name in ("exp_avg", "exp_avg_sq"):
                if name in state:
                    state[name] = state[name].to(device)
        for name, actual in (
            ("policy", self.policy.state_dict()),
            ("value", self.value.state_dict()),
            ("optimizer", self.optimizer.state_dict()["state"]),
        ):
            if tensor_state_digest(actual) != summary[f"{name}_state_sha256"]:
                raise RuntimeError(f"Native load failed to restore {name} tensors exactly")
        self._resume_offset = self.run_state.environment_steps
        self._invocation_start_updates = self.run_state.updates
        self.last_saved_update = self.run_state.updates
        self.verify_state()
        if self.device.type == "cuda":
            summary["load_peak_allocated_gib"] = torch.cuda.max_memory_allocated(self.device) / 2**30
        if self.encoder is not None:
            self.encoder.restore()
        summary["load_cpu_peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        summary["loaded_head_sha256"] = module_digest(self.policy)
        return summary

    def _before_optimizer_step(self, optimizer: torch.optim.Optimizer, args: tuple, kwargs: dict) -> None:
        norms = []
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    self._finite_gradients &= bool(torch.isfinite(parameter.grad).all())
                    norms.append(torch.linalg.vector_norm(parameter.grad.float()))
        self._gradient_norm = float(torch.linalg.vector_norm(torch.stack(norms)))
        if not self._finite_gradients:
            raise FloatingPointError("Native PPO produced non-finite gradients")

    def _after_optimizer_step(self, optimizer: torch.optim.Optimizer, args: tuple, kwargs: dict) -> None:
        self.run_state.optimizer_steps += 1
