# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Assemble original-head policy, native skrl PPO/RandomMemory/SequentialTrainer."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from importlib.metadata import version
from pathlib import Path

import gr00t
import torch
import torch.distributed as dist
from skrl.agents.torch import ExperimentCfg
from skrl.agents.torch.ppo import PPO_CFG
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer

from scripts.reinforcement_learning.gr00t_skrl.agent import Gr00tPPO, RunState, validate_checkpoint
from scripts.reinforcement_learning.gr00t_skrl.distributed import (
    check_rank_config,
    gather_rank_objects,
    measure_phase,
    rank_seed,
    require_rank_agreement,
    runner_digest,
    seed_rng,
)
from scripts.reinforcement_learning.gr00t_skrl.environment import RemoteEnvironment
from scripts.reinforcement_learning.gr00t_skrl.policy import ChainPolicy, FrozenEncoder, StateCritic, module_digest
from scripts.reinforcement_learning.gr00t_skrl.protocol import RunConfig


def checkpoint_metadata(cfg: RunConfig, encoder: FrozenEncoder) -> dict:
    """Describe the versioned probability contract and immutable model/processor inputs."""
    versions = {package: version(package) for package in ("skrl", "torch", "numpy", "transformers", "tensorboard")}
    if versions["skrl"] != "2.1.0":
        raise ValueError("This runner was validated against skrl 2.1.0")
    processor_files = {}
    for filename in (
        "config.json",
        "processor_config.json",
        "statistics.json",
        "embodiment_id.json",
        "model.safetensors.index.json",
    ):
        processor_files[filename] = hashlib.sha256((Path(cfg.model_path) / filename).read_bytes()).hexdigest()
    return {
        "policy_version": "gr00t-n17-full-chain-v1",
        "model_path": cfg.model_path,
        "backbone_path": cfg.backbone_path,
        "processor_files": processor_files,
        "generation_steps": cfg.generation_steps,
        "sigma": cfg.sigma,
        "horizon": encoder.head.config.action_horizon,
        "action_dim": encoder.head.config.max_action_dim,
        "feature_layout": vars(encoder.layout),
        "dependencies": versions,
        "instruction": cfg.instruction,
        "task": cfg.task,
        "backend": "isaacsim_physx",
        "learning_rate": cfg.learning_rate,
        "rollouts": cfg.rollouts,
        "learning_epochs": cfg.learning_epochs,
        "training_batch_size": cfg.training_batch_size,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ratio_clip": 0.2,
        "probability": "sum over K*H*D Gaussian transition log densities; x_0 constant omitted",
    }


def run(cfg: RunConfig) -> None:
    """Run a complete native rollout, checkpoint or inference, with reproducible diagnostics."""
    torch.cuda.set_device(cfg.model_device)
    topology = check_rank_config(cfg)
    seed_rng(rank_seed(cfg.seed, cfg.rank))
    torch.cuda.reset_peak_memory_stats(cfg.model_device)
    encoder = FrozenEncoder(cfg, device=cfg.model_device)
    policy = ChainPolicy(
        encoder.head,
        encoder.layout,
        cfg.generation_steps,
        cfg.sigma,
        cfg.model_device,
        sampling_batch_size=cfg.sampling_batch_size,
    )
    if not all(parameter.requires_grad for parameter in policy.parameters()):
        raise RuntimeError("The complete original action head must remain trainable")
    critic = StateCritic(policy.observation_space, policy.action_space, cfg.model_device)
    metadata = checkpoint_metadata(cfg, encoder)
    run_state = RunState(metadata, cfg=cfg, topology=topology)
    backbone_before = module_digest(encoder.backbone)
    if cfg.distributed:
        model_code = hashlib.sha256()
        for path in sorted(Path(gr00t.__file__).parent.rglob("*.py")):
            model_code.update(str(path.relative_to(Path(gr00t.__file__).parent)).encode())
            model_code.update(path.read_bytes())
        require_rank_agreement(
            "assets/code/dependencies",
            {
                "metadata": metadata,
                "backbone_sha256": backbone_before,
                "initial_head_sha256": module_digest(policy),
                "runner_sha256": runner_digest(),
                "model_code_sha256": model_code.hexdigest(),
            },
        )
    resume_summary = validate_checkpoint(cfg.resume, metadata) if cfg.resume else None
    require_rank_agreement("checkpoint identity", resume_summary)
    agent_cfg = PPO_CFG(
        rollouts=cfg.rollouts,
        learning_epochs=cfg.learning_epochs,
        mini_batches=cfg.mini_batches,
        discount_factor=0.99,
        gae_lambda=0.95,
        learning_rate=cfg.learning_rate,
        grad_norm_clip=1.0,
        ratio_clip=0.2,
        value_clip=0.2,
        entropy_loss_scale=0.0,
        time_limit_bootstrap=True,
        mixed_precision=False,
        experiment=ExperimentCfg(
            directory=cfg.run_dir, experiment_name="native", write_interval=cfg.rollouts, checkpoint_interval=0
        ),
    )
    memory = RandomMemory(memory_size=cfg.rollouts, num_envs=cfg.num_envs, device=cfg.model_device)
    # Native skrl broadcasts state_dict objects; leave GPU headroom for its temporary tensors.
    if cfg.distributed:
        encoder.offload()
    agent = Gr00tPPO(
        encoder=encoder,
        run_state=run_state,
        diagnostic_batch_size=cfg.sampling_batch_size,
        checkpoint_interval=cfg.checkpoint_interval,
        verify_distributed_state=cfg.verify_distributed_state,
        warmup_updates=cfg.warmup_updates,
        models={"policy": policy, "value": critic},
        memory=memory,
        observation_space=policy.observation_space,
        state_space=critic.state_space,
        action_space=policy.action_space,
        device=cfg.model_device,
        cfg=agent_cfg,
    )
    if cfg.distributed:
        encoder.restore()
    env = RemoteEnvironment(cfg, encoder, policy)
    try:
        trainer = SequentialTrainer(
            env=env,
            agents=agent,
            cfg={
                "timesteps": cfg.timesteps,
                "headless": True,
                "disable_progressbar": True,
                "close_environment_at_exit": False,
                "stochastic_evaluation": True,
            },
        )
        resume_summary = agent.resume(cfg.resume, summary=resume_summary) if cfg.resume else None
        if cfg.verify_distributed_state and cfg.resume is None:
            agent.verify_state()
        # Verify the real model and native memory interface before any parameter updates.
        observations, _ = env.reset()
        consistency = 0.0
        with torch.no_grad():
            for batch in observations.split(cfg.sampling_batch_size):
                chain, output = policy.act({"observations": batch})
                _, recomputed = policy.act({"observations": batch, "taken_actions": chain})
                consistency = max(consistency, float((output["log_prob"] - recomputed["log_prob"]).abs().max()))
        if consistency > 0.01:
            raise RuntimeError(f"Sampling/recompute likelihood mismatch: {consistency}")
        sampling_peak = torch.cuda.max_memory_allocated() / 2**30
        print(
            json.dumps(
                {
                    "metadata": metadata,
                    "trainable_head_parameters": sum(p.numel() for p in policy.parameters() if p.requires_grad),
                    "backbone_trainable_parameters": sum(
                        p.numel() for p in encoder.backbone.parameters() if p.requires_grad
                    ),
                    "chain_size": policy.num_actions,
                    "num_envs": cfg.num_envs,
                    "rollout_samples": cfg.rollouts * cfg.num_envs,
                    "global_rollout_samples": cfg.world_size * cfg.rollouts * cfg.num_envs,
                    "rank": cfg.rank,
                    "world_size": cfg.world_size,
                    "model_device": cfg.model_device,
                    "mini_batches": cfg.mini_batches,
                    "observation_memory_gib": cfg.rollouts * cfg.num_envs * encoder.layout.size * 4 / 2**30,
                    "log_prob_recompute_error": consistency,
                    "resume": resume_summary,
                }
            ),
            flush=True,
        )
        if resume_summary is not None:
            # The diagnostic above consumed RNG; restore again immediately before native trainer interaction.
            run_state.restore_rng()
        agent.phase_seconds.clear()
        env.phase_seconds.clear()
        start_counters = run_state.counters()
        if dist.is_initialized():
            dist.barrier()
        torch.cuda.synchronize(cfg.model_device)
        start_time = time.monotonic()
        if cfg.mode == "train":
            trainer.train()
        else:
            trainer.eval()
        if cfg.mode == "train" and agent.last_saved_update != run_state.updates:
            with measure_phase(agent.phase_seconds, "backbone_offload", agent.device):
                encoder.offload()
            try:
                agent.save_boundary()
            finally:
                with measure_phase(agent.phase_seconds, "backbone_restore", agent.device):
                    encoder.restore()
        if dist.is_initialized():
            dist.barrier()
        torch.cuda.synchronize(cfg.model_device)
        invocation_seconds = time.monotonic() - start_time
        interaction_seconds = time.monotonic() - (agent.measurement_start or start_time)
        measurement_counters = agent.measurement_counters or start_counters
        measured_global_transitions = run_state.transitions - measurement_counters["transitions"]
        backbone_unchanged = module_digest(encoder.backbone) == backbone_before
        if not backbone_unchanged or any(p.grad is not None for p in encoder.backbone.parameters()):
            raise RuntimeError("Frozen backbone changed or received gradients")
        metrics = {
            "metadata": metadata,
            "seed": cfg.seed,
            "log_prob_recompute_error": consistency,
            "peak_cpu_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
            "sampling_peak_allocated_gib": sampling_peak,
            "final_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "backbone_unchanged": backbone_unchanged,
            "backbone_sha256": backbone_before,
            "resume": resume_summary,
            "updates": agent.diagnostics,
            "transitions": env.metrics,
            "environment_steps": run_state.environment_steps,
            "transitions_count": run_state.transitions,
            "num_envs": cfg.num_envs,
            "rank": cfg.rank,
            "world_size": cfg.world_size,
            "new_local_transitions": cfg.timesteps * cfg.num_envs,
            "new_global_transitions": run_state.transitions - start_counters["transitions"],
            "measured_global_transitions": measured_global_transitions,
            "measured_local_transitions": measured_global_transitions // cfg.world_size,
            "warmup_updates": cfg.warmup_updates,
            "phase_seconds": {**env.phase_seconds, **agent.phase_seconds},
            "interaction_seconds": interaction_seconds,
            "invocation_seconds": invocation_seconds,
            "transitions_per_second": measured_global_transitions / interaction_seconds,
            "observation_memory_gib": cfg.rollouts * cfg.num_envs * encoder.layout.size * 4 / 2**30,
            "optimizer_steps": run_state.optimizer_steps,
        }
        for env_id in range(cfg.num_envs):
            if not any(item["motion_m"] > 1e-6 for item in env.metrics if item["env_id"] == env_id):
                raise RuntimeError(f"No measured robot motion in environment {env_id}")
        if not all(item["table_rgb_std"] > 1 and item["wrist_rgb_std"] > 1 for item in env.metrics):
            raise RuntimeError("Invalid or constant RGB camera observations")
        (Path(cfg.run_dir) / "metrics.json").write_text(json.dumps(metrics, indent=2))
        require_rank_agreement("final counters", run_state.counters())
        ranks = gather_rank_objects(
            {
                "rank": cfg.rank,
                "interaction_seconds": interaction_seconds,
                "phase_seconds": metrics["phase_seconds"],
                "new_local_transitions": metrics["new_local_transitions"],
                "measured_local_transitions": metrics["measured_local_transitions"],
                "sampling_peak_allocated_gib": sampling_peak,
                "final_peak_allocated_gib": metrics["final_peak_allocated_gib"],
                "backbone_unchanged": backbone_unchanged,
            }
        )
        if cfg.rank == 0 and cfg.job_dir is not None:
            window = max(item["interaction_seconds"] for item in ranks)
            (Path(cfg.job_dir) / "metrics.json").write_text(
                json.dumps(
                    {
                        **run_state.counters(),
                        "world_size": cfg.world_size,
                        "new_global_transitions": metrics["new_global_transitions"],
                        "measured_global_transitions": measured_global_transitions,
                        "warmup_updates": cfg.warmup_updates,
                        "interaction_seconds": window,
                        "transitions_per_second": measured_global_transitions / window,
                        "ranks": ranks,
                    },
                    indent=2,
                )
            )
        print(
            json.dumps(
                {
                    "environment_steps": run_state.environment_steps,
                    "transitions": run_state.transitions,
                    "optimizer_steps": run_state.optimizer_steps,
                    "backbone_unchanged": backbone_unchanged,
                }
            ),
            flush=True,
        )
    finally:
        env.close()
        if hasattr(agent, "writer"):
            agent.writer.close()


def main() -> None:
    """Read the launcher-owned configuration in the GR00T uv environment."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = RunConfig(**json.loads(Path(args.config).read_text()))
    cfg.validate()
    try:
        run(cfg)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
