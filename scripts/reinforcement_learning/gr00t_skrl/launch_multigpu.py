# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch one isolated GR00T simulation/model pair per GPU through Isaac Lab torchrun."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.cli.multigpu import MultiGpuLauncherCfg, parse_launcher_args, run_multigpu_cli

from scripts.reinforcement_learning.gr00t_skrl.launch import REPO_ROOT


def main(argv: list[str] | None = None) -> int:
    """Validate a fixed-size job, reserve node output and delegate to the shared launcher.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.

    Returns:
        Launcher exit code.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    cfg = MultiGpuLauncherCfg(
        prog="gr00t_skrl.launch_multigpu",
        description=__doc__,
        worker_script=str(Path(__file__).with_name("launch.py").absolute()),
        supports_jax=False,
    )
    args, worker_args = parse_launcher_args(argv, cfg)
    worker_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    worker_parser.add_argument("--job_id")
    worker_parser.add_argument("--run_dir")
    worker_parser.add_argument("--rpc_timeout", type=float)
    worker_parser.add_argument("--checkpoint_interval", type=int)
    worker, remaining = worker_parser.parse_known_args(worker_args)
    if args.max_restarts not in (None, "0"):
        worker_parser.error("Automatic restarts are unsupported; restart explicitly with --resume")
    if ":" in str(args.nnodes):
        worker_parser.error("Elastic node ranges are unsupported; specify a fixed --nnodes")
    if args.nproc_per_node in ("cpu", "auto"):
        worker_parser.error("GR00T requires --num_gpus gpu or a positive GPU count")
    if args.nproc_per_node != "gpu" and int(args.nproc_per_node) < 1:
        worker_parser.error("--num_gpus must be positive")
    node_count = int(args.nnodes or 1)
    if node_count < 1:
        worker_parser.error("--nnodes must be positive")
    if not 0 <= int(args.node_rank or 0) < node_count:
        worker_parser.error("--node_rank must be within the fixed node count")
    if node_count > 1 and (not worker.job_id or not worker.run_dir or args.node_rank is None):
        worker_parser.error("Multi-node jobs require the same --job_id, --run_dir and an explicit --node_rank")
    if node_count > 1 and not (args.master_addr or args.rdzv_endpoint):
        worker_parser.error("Multi-node jobs require --master_addr or --rdzv_endpoint")
    if args.rdzv_id is not None and worker.job_id is not None and args.rdzv_id != worker.job_id:
        worker_parser.error("--rdzv_id must match --job_id")
    job_id = worker.job_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path(worker.run_dir or REPO_ROOT / "logs/gr00t_skrl" / job_id).absolute()
    if not args.dry_run:
        node_dir = run_dir / "nodes"
        node_dir.mkdir(parents=True, exist_ok=True)
        # Exclusive per-node claims work with both shared and node-local filesystems.
        with (node_dir / f"node_{int(args.node_rank or 0):05d}.json").open("x") as stream:
            json.dump({"job_id": job_id, "nnodes": node_count, "argv": argv}, stream, indent=2)
    os.environ["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
    launcher_args = ["--num_gpus", str(args.nproc_per_node)]
    for name, value in vars(args).items():
        if name in ("nproc_per_node", "dry_run", "log_all_ranks") or value is None:
            continue
        launcher_args.extend((f"--{name}", str(value)))
    if args.dry_run:
        launcher_args.append("--dry_run")
    if args.log_all_ranks:
        launcher_args.append("--log_all_ranks")
    if args.max_restarts is None:
        launcher_args.extend(("--max_restarts", "0"))
    if args.rdzv_id is None:
        launcher_args.extend(("--rdzv_id", job_id))
    launcher_args.extend(
        (
            "--",
            *remaining,
            "--run_dir",
            str(run_dir),
            "--job_id",
            job_id,
            "--rpc_timeout",
            str(worker.rpc_timeout if worker.rpc_timeout is not None else 1800),
            "--checkpoint_interval",
            str(worker.checkpoint_interval if worker.checkpoint_interval is not None else 10),
        )
    )
    return run_multigpu_cli(launcher_args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
