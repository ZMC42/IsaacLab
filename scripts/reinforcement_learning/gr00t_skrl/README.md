# GR00T N1.7 with native skrl PPO

This local runner fine-tunes the **complete original N1.7 action head** from the existing micro-SFT checkpoint. It uses skrl 2.1.0 PyTorch `PPO`, `RandomMemory` and `SequentialTrainer` for rollout collection, GAE, losses, backward passes, Adam updates, TensorBoard and checkpoint serialization. The frozen vision/language backbone is held separately from the policy, optimizer and checkpoint.

The simulator and model run in separate uv environments and processes, with one pair per GPU. The simulator uses Isaac Sim 6.1 / PhysX, one Franka per environment and the task's table/wrist RGB cameras. This is an engineering smoke test with a local approach reward; it does not establish stacking ability or PPO convergence. Shared task definitions and Isaac-GR00T source are unchanged.

## Parallel environment training

Pass `--num-envs N` (alias `--num_envs`) to create N environments in the existing simulation
process. One model and one PPO agent control all environments. Every step encodes their
observations, samples independent full generation chains, decodes each with its own physical
reference state, and advances all robots together. Deterministic resets are preserved; each
environment resets independently on termination or timeout.

```bash
uv run --no-sync python -m scripts.reinforcement_learning.gr00t_skrl.launch \
  --num-envs 2 --timesteps 4 --rollouts 2 --episode_steps 2 \
  --encoder_batch_size 1 --sampling_batch_size 1 --training_batch_size 1
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--num-envs` | 1 | Number of physical environments |
| `--encoder_batch_size` | 1 | Maximum frozen-backbone encoding batch |
| `--sampling_batch_size` | 1 | Maximum action-generation and likelihood-diagnostic batch |
| `--training_batch_size` | 1 | Maximum native PPO minibatch |
| `--timesteps` | 2 | Synchronous steps in this invocation; collects `timesteps × N` transitions |
| `--rollouts` | 2 | Steps per environment before each update; memory holds `rollouts × N` samples |

Training uses `ceil(rollouts × N / training_batch_size)` native minibatches per epoch.
Uneven batches cover every sample without replacement. There is no gradient accumulation;
increasing N at a fixed batch size increases optimizer steps per update. The example above
collects eight transitions, runs two PPO updates and, with two learning epochs, takes 16 Adam
steps. Learning rate and probability semantics are unchanged.

RPC images are `[N, height, width, 3]`, state and terminal state are `[N, 8]`, controls are
`[N, 7]`, and skrl reward/done tensors are `[N, 1]`. Native memory retains `[rollouts, N, ...]`
until PPO samples minibatches. Timeout bootstrap uses only that environment's pre-reset state;
true termination takes precedence over timeout. The single-environment trainer reset request
is still consumed without resetting the simulation again.

All batch sizes default to one to limit activation memory. GPU rollout storage still grows with
`rollouts × N`: each padded FP32 observation is about 8 MiB for the default 1024 tokens and
2048-dimensional backbone. Native randomized minibatches also allocate sampled tensor copies.
Rendering memory grows with the number of camera views. Start with N=2 and measure before raising
N or any batch size. Logs report the observation-storage estimate; `metrics.json` records
per-environment commands, motion, rewards and camera variation, total `transitions_count`,
`transitions_per_second` (including updates/checkpoint writes, excluding initialization), and
model memory peaks. `processes.json` records sampled whole-GPU memory.

`RunState.environment_steps`, TensorBoard's step axis and checkpoint names continue to count
synchronous steps. The new cumulative `RunState.transitions` counts physical transitions even
when N changes after resuming. A legacy single-environment checkpoint defaults its missing
transition counter to `environment_steps` and its missing training batch size to one. Resume
allows changing N and encoder/sampling batch sizes; model/processor/probability and PPO settings,
including training batch size, must match. Physics starts from a fresh episode, so changing N or
batching does not promise the same sampled trajectory. Use `--resume PATH` with the same model
asset paths recorded by the original run. `--mode inference --resume PATH --num-envs N` also
supports parallel inference without optimizer updates.

The following two-step runs passed on an RTX 4090 (24 GB) on 2026-10-13, with all three batch
sizes set to one, two learning epochs and a checkpoint write:

| Run | New transitions | New Adam steps | Sampled whole-GPU peak | Transitions/s |
| --- | ---: | ---: | ---: | ---: |
| N=1, fresh | 2 | 4 | 20,796 MiB | 0.127 |
| N=2, resumed from legacy N=1 | 4 | 8 | 21,174 MiB | 0.246 |
| N=4, fresh | 8 | 16 | 21,232 MiB | 0.426 |

All robots moved, all camera views varied, likelihood recomputation error was zero, gradients
were finite, and frozen-backbone fingerprints matched. The real partial-reset test also passed.
These short engineering runs include checkpoint I/O and use different rollout sizes and starting
weights; the rates do not establish sustained-training speedup or convergence.
Restoring the N=4 checkpoint into N=2 inference with encoder/sampling batch sizes of two also
passed: four new transitions were collected, likelihood recomputation error was zero, and the
optimizer stayed at step 16.

## Multi-GPU and multi-node training

`launch_multigpu` reuses Isaac Lab's `MultiGpuLauncherCfg` / `run_multigpu_cli` and runs
`launch.py` once per torchrun rank. Every GPU owns a simulator/model pair and a private local
socket. Only model processes inherit the rank/rendezvous environment and join skrl's native
NCCL group. Models use `cuda:LOCAL_RANK`; each simulator sees only that GPU as `cuda:0`.
The supervisor never imports skrl. Both children retain their existing uv project and `--no-sync`.
The existing single-GPU `launch` command and its defaults remain available.

Replace the asset paths below with the same absolute paths on every node. The GR00T uv project
defaults to `../Isaac-GR00T`; override `--model_project` if necessary.

```bash
uv run --no-sync python -m scripts.reinforcement_learning.gr00t_skrl.launch_multigpu \
  --num_gpus 2 --job_id gr00t_two_gpu --run_dir /data/runs/gr00t_two_gpu \
  --model_path /data/models/stack-cube-n1.7-sft/checkpoint \
  --backbone_path /data/models/nvidia/Cosmos-Reason2-2B \
  --num_envs 2 --timesteps 4 --rollouts 2 --episode_steps 2 \
  --checkpoint_interval 1 --verify_distributed_state
```

Add `--dry_run` to print the command without reserving output directories or starting workers.
The launcher also accepts Isaac Lab's torchrun logging options, including `--log_all_ranks`,
`--tee`, `--redirects` and `--log_dir`. Console output defaults to local rank 0 on each node;
every rank always keeps its own simulation/model logs and exit codes.

For two nodes, run this on **each** node with `GR00T_NODE_RANK=0` on the master and
`GR00T_NODE_RANK=1` on the other node:

```bash
uv run --no-sync python -m scripts.reinforcement_learning.gr00t_skrl.launch_multigpu \
  --num_gpus 2 --nnodes 2 --node_rank "$GR00T_NODE_RANK" \
  --master_addr 10.0.0.10 --master_port 29500 \
  --job_id gr00t_four_gpu --run_dir /data/runs/gr00t_four_gpu \
  --model_path /data/models/stack-cube-n1.7-sft/checkpoint \
  --backbone_path /data/models/nvidia/Cosmos-Reason2-2B \
  --num_envs 2 --timesteps 4 --episode_steps 2 --checkpoint_interval 1
```

The node count is fixed. `--rdzv_backend`, `--rdzv_endpoint` and `--rdzv_id` are also supported;
an explicit rendezvous ID must match `--job_id`. Elastic node ranges and nonzero `--max_restarts`
are rejected. Every multi-node invocation requires explicit `--job_id`, `--run_dir`,
`--node_rank` and a master address or rendezvous endpoint. Node output claims and rank directories
are exclusive: use a fresh run directory when restarting. If Viser is enabled, only global rank 0
starts its viewer.

All nodes must have identical code, dependency versions, task/PPO settings, model/processor assets
and resume contents at identical absolute paths. Shared storage is optional: pre-stage assets and
the checkpoint on each node when using local disks. Before constructing the native agent, the
models compare rollout settings, simulator checkout/version identity, runner/GR00T code hashes,
processor/config hashes, actual frozen-backbone and initial-head hashes, and the existing CPU
checkpoint validation summaries. The CPU summaries include optimizer settings and RunState/RNG
identity as well as model/Adam tensors. No full model or checkpoint is exchanged by these checks.

For W total ranks, N environments per rank, R rollout steps, minibatch upper bound B and E epochs:

| Quantity | Contract |
| --- | --- |
| Local/global rollout samples | `R × N` / `W × R × N` |
| Native minibatches per epoch | `ceil(R × N / B)` on every rank |
| Shared Adam steps per update | `E × ceil(R × N / B)` |
| New global transitions | `W × N × timesteps` |
| New synchronous steps | `timesteps` |

Rollouts and GAE stay local. Native skrl broadcasts initial policy/critic parameters and averages
gradients after backward, before clipping and Adam. Advantages are normalized locally, so this is
not equivalent to normalizing a centrally combined rollout. Uneven native minibatches retain every
sample; there is no padding, dropped tail or extra gradient accumulation. Expanding W grows the
global sample count and leaves the learning rate unchanged. Rank-derived seeds give independent
sampling/shuffle streams; deterministic physics resets are preserved.

### Checkpoints, recovery and failure

`--checkpoint_interval` counts cumulative **PPO updates**, defaults to 1 for `launch` and 10 for
`launch_multigpu`, and must be positive. Normal training completion saves the last full update if
it was not already saved. Inference does not write checkpoints. Every save checks shared counters,
collects Python/NumPy/Torch CPU/current-GPU RNG for every rank, and lets only global rank 0 call
native `save`. It writes a temporary file in the destination directory, fsyncs, atomically renames
it, then communicates the result to all ranks. Temporary files are never resume candidates.
`--verify_distributed_state` additionally compares full policy/critic/Adam hashes at save boundaries
and initial construction; use it for correctness runs because these hashes add diagnostic cost.
All resumes verify loaded states across ranks regardless of this option.

The four top-level checkpoint modules and original head state-dict names are unchanged.
Versioned distribution metadata and per-rank RNG are stored inside `run_state`, with the original
`rng` field retained. Resume accepts old single-environment and single-GPU checkpoints and allows
changing GPU/node count, rank mapping, environment count, encoding/sampling batches and save
interval. Immutable model/processor/chain/PPO settings, including training batch size, still must
match. Historical transitions are never recalculated using the new topology.

Matching topology and N restore the corresponding rank's RNG. A topology/N change derives new
streams from the saved base seed, completed updates and new rank. Legacy checkpoints without a
base seed use the invocation seed when deriving new streams. Physics always starts a new episode.
Single-GPU inference can read the extended checkpoints via the original `launch --mode inference`.

```text
/data/runs/<job>/
├── nodes/node_00000.json             # Exclusive node claim and launcher arguments
├── ranks/rank_00000/                 # Global rank, even with node-local storage
│   ├── config.json, simulation_versions.json
│   ├── simulation.log, train.log, metrics.json, processes.json
│   └── native/                      # TensorBoard and checkpoints only on rank 0
│       └── checkpoints/agent_<synchronous_steps>.pt
├── ranks/rank_00001/...
└── metrics.json                     # Rank 0's shared counters and small rank summaries
```

With local disks, copy the rank-0 checkpoint to the same resume path on every node before restarting.
Any child failure terminates its local pair; torchrun then aborts the remaining ranks. SIGINT/SIGTERM
go through local process-group and socket cleanup, including descendants whose uv leader has exited.
Interrupted or failed updates are not saved. Storage failure fails the whole job and leaves earlier
checkpoints intact. Restart explicitly with `--resume` and a new job directory.

RPC timeouts default to 1800 s in the multi-GPU entry and 300 s in the original entry; override
`--rpc_timeout` for large asset loads or debugging. The native skrl NCCL group uses PyTorch's finite
default timeout (600 s in the validated model environment); torchrun rendezvous also has a finite
timeout. Actual multi-node fault propagation and cleanup still need hardware validation.

### Throughput and validation status

`--warmup_updates K` excludes the first K complete updates of this invocation from the synchronized
throughput window and phase totals; it defaults to 0 and must leave at least one measured update.
Warmup does not discard samples or change counters/save scheduling. `metrics.json` reports both all
new transitions and measured transitions, plus time spent in simulation/RPC, encoding, sampling,
native PPO (including synchronization), diagnostics, checkpoint and backbone offload/restore.
The main rate includes periodic/final checkpoint writes and diagnostics; initialization and the
final frozen-backbone verification are outside the window. CUDA work is synchronized for timing.
The job rate uses measured global transitions divided by the slowest rank's window, not the sum of
rank rates. Rank logs retain invocation-local counts separately from cumulative historical counts.

For sustained comparisons, use the same initial weights and per-rank N/R/B/E, chain configuration
and save interval (start with 10), several warmup updates, and a window covering multiple save
cycles. Repeat measurements; report `speedup(W) = throughput(W) / throughput(1)` and
`efficiency(W) = speedup(W) / W`, phase bottlenecks and whole-GPU peaks. Run an independent NCCL
probe before real two-GPU training. Do not add NCCL workarounds unless that probe demonstrates a need.
Native object broadcast and contiguous gradient buffers increase temporary memory beyond the
earlier single-GPU measurements; those peaks are not a multi-GPU memory budget.

On the local single RTX 4090, the new torchrun entry was exercised with one rank and N=2, restoring
the old `smoke_v3` checkpoint: four new synchronous steps produced eight transitions and two PPO
updates (16 new Adam steps), followed by the final interval-10 checkpoint. Both children exited
with zero status, gradients/losses were finite, and the frozen backbone stayed unchanged. This is
single-rank compatibility evidence. Real two-GPU NCCL, multi-node training/resume/failure tests and
sustained speedup have **not** been validated on this one-GPU host.

The extended checkpoint also passed original-entry N=1 inference: two new transitions, unchanged
Adam step 20 and no checkpoint writes. A subsequent one-rank N=2 run passed periodic saving and
warmup accounting: 12 new transitions, eight measured after one warmup update, cumulative Adam
step 44, and checkpoints at synchronous steps 8 and 12. Both children exited normally. These
short runs validate accounting and persistence, not sustained throughput.
Single-rank torchrun fault injection during initialization also passed: killing either child and
interrupting the launcher returned nonzero status, recorded both child exit codes, removed the
socket directory and left no live process in either child group. These checks cover local startup
cleanup; faults inside real multi-rank PPO/NCCL collectives still require dedicated hardware runs.

## Debugging and learning

Open the Isaac Lab root in VS Code with the Python and Python Debugger extensions installed on
the remote host. Select **GR00T PPO: 两步训练（自动调试模型与仿真）** in
[launch.json](../../../.vscode/launch.json) and press F5. The configuration automatically attaches
both Python children, runs one two-step rollout and extends RPC deadlines to two hours for
breakpoints. Resume and inference configurations prompt for a native checkpoint's absolute path.
Each launch creates a fresh output directory.

The debug-only `--debug_subprocesses` option invokes each project's existing `.venv/bin/python`
directly, allowing VS Code's `subProcess` injection to reach both environments without following
uv's Rust process. It does not install dependencies or change the training loop. Normal launches
still use `uv run --no-sync`. Child output remains in `train.log` and `simulation.log`.

See the Chinese [RL pipeline learning guide](../../../docs/RL_PIPELINE_GUIDE.md) for the theory,
code map, breakpoint walkthrough, tensor shapes and checkpoint workflow.

## Viser on a headless SSH server

Pass `--viser` to this runner's launcher to view the live Franka scene in your local browser.
PhysX still runs the simulator; Viser consumes its scene
data without opening a desktop window. Install the optional viewer in the Isaac Lab environment
while retaining existing packages, then launch:

```bash
uv sync --inexact --extra viser
uv run --no-sync python -m scripts.reinforcement_learning.gr00t_skrl.launch \
  --viser --timesteps 8 --rollouts 2 --rpc_timeout 7200
```

In VS Code Remote SSH, forward **8080** in the **Ports** panel and open
`http://localhost:8080` on your local machine. Use `--viser_port PORT` to select a different
server port. For models outside the default sibling layout, pass `--model_path`,
`--backbone_path` and `--model_project` as documented below. Add `--viser` to a VS Code launch
configuration's `args` to visualize a debugging session too.

The viewer keeps rendering while waiting for the model without advancing physics. During model
loading, inference, PPO optimization and checkpoint writing, the robot can remain stationary.
The table and wrist RGB cameras remain the model's observations; the Viser canvas shows the 3D
scene rather than a camera stream. Pausing simulation
in the viewer also blocks the model's next simulation RPC; keep `--rpc_timeout` long enough
for an interactive pause. The viewer closes when the run exits. Each two-step PPO update
writes a checkpoint of approximately 9.1 GiB with the default rollout settings.

## Installation

Run from the Isaac Lab checkout. The existing environments are `.venv` and `../Isaac-GR00T/.venv`. Preserve the existing model dependencies when adding skrl:

```bash
uv --no-config pip freeze --python ../Isaac-GR00T/.venv/bin/python \
  --exclude-editable > /tmp/gr00t-skrl-existing-constraints.txt
uv --no-config pip install --python ../Isaac-GR00T/.venv/bin/python \
  --constraint /tmp/gr00t-skrl-existing-constraints.txt \
  -r scripts/reinforcement_learning/gr00t_skrl/requirements-model.txt
```

`--no-config` is necessary here: Isaac Lab's uv configuration overrides NumPy to version 2+, whereas the isolated GR00T environment requires NumPy 1.26.4. Runtime commands use `--no-sync` so that the two environments keep their separate dependency versions.

Validated model versions: Torch 2.9.0+cu128, NumPy 1.26.4, Transformers 4.57.3, skrl 2.1.0 and TensorBoard 2.20.0. Simulator versions are recorded separately with each run. The default inputs are:

- `../embodied-template/models/stack-cube-n1.7-sft/checkpoint`
- `../embodied-template/models/nvidia/Cosmos-Reason2-2B`

Keep `nvidia/Cosmos-Reason2` in the backbone path, including its symlink name; the upstream model selects its backbone class from this string. Model assets must already be local. The launcher sets `HF_HUB_OFFLINE=1` and the previously accepted `OMNI_KIT_ACCEPT_EULA=YES`.

## One-command cloud deployment

The deployment script targets **Ubuntu 22.04/24.04 x86_64** with a rendering-capable NVIDIA GPU
(24 GB VRAM or more), at least 32 GB host RAM and driver **580.65.06 or newer** for the simulator's
CUDA 13.0 Torch build. Driver 580.95.05+ is recommended by the Isaac Lab installation guide.
CUDA availability alone does not establish Isaac Sim rendering support; use a GPU supported by
[Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html).
Allow at least **100 GiB free** for environments, download caches and smoke-test outputs in addition
to the transferred models; each PPO checkpoint adds about 9.1 GiB. The script checks the checkout,
GR00T and uv cache volumes. It uses the provider's installed driver; CUDA runtime libraries come
with the Python wheels, so a separate CUDA toolkit is unnecessary for this installation.

Transfer this Isaac Lab branch and both model directories to the server first. The micro-SFT
checkpoint is a local training artifact; the script does not substitute a public base model.
Do **not** copy `.venv` directories between servers. When transferring the backbone, dereference
its symlink (for example, use `rsync -aL`) or also transfer its target. Keep the destination name
`nvidia/Cosmos-Reason2-2B`. With the default sibling directory layout:

```text
workspace/
├── IsaacLab/                 # this branch, including uv.lock and deploy.sh
├── Isaac-GR00T/              # script creates this if absent
└── embodied-template/models/
    ├── stack-cube-n1.7-sft/checkpoint/
    └── nvidia/Cosmos-Reason2-2B/
```

Run from the transferred Isaac Lab checkout:

```bash
bash scripts/reinforcement_learning/gr00t_skrl/deploy.sh
```

It installs OS packages with root/passwordless sudo, bootstraps uv if absent, obtains Python 3.12,
fetches the validated GR00T commit `d2b7e75b937e3ec9aa5dbc798f08b89692c49734`, and installs the
two `.venv` environments from their frozen lockfiles. GR00T sync runs from its own directory;
the additional PPO dependencies use `uv --no-config pip` with a snapshot constraint file.
Existing extra packages are retained by `--inexact`. A different or modified existing GR00T
checkout is rejected; pass a new `--model_project` to keep it separate. The deployment is
noninteractive and uses the previously accepted `OMNI_KIT_ACCEPT_EULA=YES`.

Default verification checks dependency versions, both CUDA environments, local weight shards,
offline processor loading and a falling procedural sphere in headless Kit/PhysX using the task's
simulation config. It prints the launch command with your paths and GPU. The processor uses
`--backbone_path` even when its saved config contains a previous server's absolute path; saved
checkpoint files and processor fingerprints remain unchanged.

For custom locations, a preconfigured image, and full two-step PPO verification:

```bash
bash scripts/reinforcement_learning/gr00t_skrl/deploy.sh \
  --model_project /data/Isaac-GR00T \
  --model_path /data/models/stack-cube-n1.7-sft/checkpoint \
  --backbone_path /data/models/nvidia/Cosmos-Reason2-2B \
  --gpu 0 --skip_system_deps --smoke_test
```

`--skip_system_deps` requires the OS libraries listed in the script to be installed already.
`--smoke_test` exercises the real Franka scene, both RGB cameras, two rollout steps, native PPO
updates and checkpoint writing. It requires network access to the task's USD assets or an existing
asset cache. Installation also needs access to Ubuntu package repositories, GitHub, PyPI, NVIDIA
and PyTorch wheel indexes. Set your proxy and `UV_CACHE_DIR` before running when needed; frozen
lockfiles retain their recorded download URLs.

Other useful invocations:

```bash
# Inspect the deployment plan without installing or launching anything.
bash scripts/reinforcement_learning/gr00t_skrl/deploy.sh --dry_run
# Recheck existing environments without reinstalling.
bash scripts/reinforcement_learning/gr00t_skrl/deploy.sh --verify_only
```

Pass the same custom path/GPU arguments when verifying. Logs and model dependency snapshots are
written under `logs/gr00t_skrl/deploy/<timestamp>/`; optional smoke outputs live in its `smoke/`
subdirectory. Re-running resumes package installation without rewriting lockfiles or overwriting
prior run directories. A failed check exits nonzero and prevents reporting deployment success.

## Policy and physical action contract

For the model-configured horizon `H`, padded action dimension `D`, generation steps `K` and fixed `sigma=0.05`:

```text
x_0 ~ Normal(0, I)
x_(k+1) ~ Normal(x_k + velocity_theta(x_k, observation, k) / K, sigma)
log_prob = sum over k, H and D of log Normal(x_(k+1); conditional_mean_k, sigma)
```

The parameter-independent density of `x_0` is omitted; it cancels in the PPO ratio. This is the joint transition density of the complete generation chain. It includes padded coordinates and unexecuted horizon positions. It is not the marginal density of the robot command.

Native memory stores flattened `x_0 ... x_K`. Teacher-forced likelihood uses these exact saved actions. The current model has `H=40`, `D=132`; `K=2` therefore stores 15,840 FP32 scalars per action. Dimensions are read from the model config. Dropout is disabled in both sampling and learning, with autograd retained. Backbone features, masks and normalized processor state are padded into a fixed observation (`--max_tokens`, default 1024); overflow raises an error. Trainable state encoders, VL normalization/self-attention and projectors are recomputed during learning.

Each environment step predicts a new chain, decodes `x_K` using the original processor and executes only its first seven-dimensional command. Arm values are clipped to `[-0.1, 0.1]`, then passed to the unchanged task action with `scale=0.5`; the runner does not multiply them again. The gripper command is binary. The processor's legacy `roll/pitch/yaw` state names contain a **principal rotation vector [rad]**, alongside XYZ [m] and two finger positions [m]. The task's signed second-finger coordinate is preserved, matching the micro-SFT processor statistics.

The local native reset event restores initial scene states and joint targets, disabling randomization only in this runner. The local reward is `step_dt * (1 - tanh(distance_to_red_cube / 0.1))`, using actual scene geometry; native reward-manager integration supplies `step_dt`. Task success, failure and timeout terms are inherited.

The state critic is `[8, 64, 1]`. With `compute_final_obs=True`, each timeout passes its reset-before-terminal critic state to native PPO bootstrapping. Simultaneous true termination and timeout does not bootstrap. The next interaction uses normal reset-after-terminal observations. The wrapper consumes the native trainer's single-environment reset request without resetting the already-reset simulation a second time.

## Training, checkpoint and inference

The real two-step smoke command succeeded on the local RTX 4090:

```bash
uv run --no-sync python -m scripts.reinforcement_learning.gr00t_skrl.launch \
  --run_dir logs/gr00t_skrl/smoke_v3 --episode_steps 2
```

It produced `logs/gr00t_skrl/smoke_v3/native/checkpoints/agent_2.pt`, completed four native Adam steps, changed the original head, preserved the frozen backbone and measured end-effector displacements of approximately 9.9 mm and 14.7 mm. Likelihood recomputation error was zero. Total GPU memory sampled once per second peaked at 20,796 MiB. Initial sampling / native update / post-update backbone restore peaked at 5.97 / 12.88 / 15.00 GiB of model-process PyTorch allocations.

The smoke used learning rate `1e-6`: post-update joint-chain ratios ranged from 0.051 to 8.15 (approximate KL 3.54, clip fraction 1.0). Losses and gradients were finite. These numbers demonstrate an update, **not a tuned or converged policy**; sustained training needs a smaller learning-rate experiment and a task reward appropriate for stacking.


The following fresh-process resume also succeeded:

```bash
uv run --no-sync python -m scripts.reinforcement_learning.gr00t_skrl.launch \
  --run_dir logs/gr00t_skrl/resume --episode_steps 2 \
  --resume logs/gr00t_skrl/smoke_v3/native/checkpoints/agent_2.pt
```

It restored head, critic and all native optimizer state tensors exactly, then advanced environment steps `2 -> 4`, updates `1 -> 2` and optimizer steps `4 -> 8`. It wrote `logs/gr00t_skrl/resume/native/checkpoints/agent_4.pt`. Model-process loading peaked at 9.08 GiB of PyTorch allocations; total sampled GPU memory peaked at 20,894 MiB. The second update retained finite losses/gradients (ratios 0.492–2.24, approximate KL 0.317). Both children exited with code zero and their sockets were removed.

Fresh-process inference succeeded without additional optimizer steps:

```bash
uv run --no-sync python -m scripts.reinforcement_learning.gr00t_skrl.launch \
  --run_dir logs/gr00t_skrl/inference --mode inference --episode_steps 2 \
  --resume logs/gr00t_skrl/resume/native/checkpoints/agent_4.pt
```

It kept optimizer step 8, validated both RGB cameras and measured approximately 11 mm of motion on each step. CPU loading peaked at 16.43 GiB RSS, model-process loading at 9.08 GiB of PyTorch GPU allocations, and the complete inference run at 17,848 MiB of sampled total GPU memory. Outputs and checkpoints are ignored artifacts, not source files. Choose a fresh `--run_dir` to reproduce these commands, or omit that option for an automatically named directory.

The launcher defaults to two environment steps, two learning epochs and two minibatches of size one (four native optimizer steps). `--episode_steps 2` explicitly exercises timeout and automatic reset. Run directories must be new; existing output is never overwritten.

A boundary checkpoint contains the original head, critic, native Adam state and a `RunState` module with cumulative counters, RNG, dependency versions, processor fingerprints and the policy contract. Automatic best-checkpoint copying is disabled. The lifecycle hook uses native `save` at full rollout boundaries and native `load` after CPU compatibility validation. CPU loading avoids a duplicate GPU copy of all checkpoint tensors. Policy, critic and optimizer tensors are independently fingerprinted before/after loading. RNG is restored after initialization and diagnostic inference; resumed physics starts from a new episode.

The head uses BF16, with FP32 chain dynamics, densities, ratios and losses. Local per-instance forward wrappers checkpoint the upstream DiT and VL transformer blocks without changing parameters or state-dict names. The backbone moves to CPU during native learning and checkpoint loading, then returns to the GPU. Native Adam uses `foreach=False`. Entropy bonus, adaptive learning rate and extra observation/value normalization are disabled. Native "Policy / Standard deviation" describes the fixed conditional transition noise.

Each output directory contains `config.json`, separate simulator/model logs, `native/` TensorBoard files and checkpoints, `metrics.json` and `processes.json`. Diagnostics include same-chain likelihood error, native losses, post-update ratios/KL/clip fraction, finite gradients, changed-head and frozen-backbone fingerprints, optimizer counters, executed controls, measured motion, RGB variation and memory peaks. PyTorch peaks describe the model process; `processes.json` samples total GPU memory once per second, including simulation. RPC has bounded messages and deadlines; child failure or interruption triggers process-group cleanup and private socket removal.

## Contract tests

These tests run entirely in the model environment; the parent test process never starts Kit:

```bash
NO_ALBUMENTATIONS_UPDATE=1 uv run --project ../Isaac-GR00T --no-sync python -m pytest \
  scripts/reinforcement_learning/gr00t_skrl/tests/test_contracts.py -q \
  --confcutdir=scripts/reinforcement_learning/gr00t_skrl/tests
```

Coverage also checks real batched NumPy 2.x ↔ 1.26 RPC compatibility, EOF and read deadlines.
Coverage protects full-chain probability/memory, original-head gradient and dropout behavior,
encoding batch order and tail batches, per-environment reference-state decoding, partial auto-reset,
independently calculated multi-environment timeout returns, complete uneven minibatch sampling,
legacy single-environment checkpoint migration and new-process restoration of model/optimizer/RNG/counters.
Two CPU/Gloo child processes test rank coordination, global counters, rank RNG restoration and
topology-change reseeding, checkpoint identity/config disagreement and atomic write failure.
Additional coverage checks multi-node dry runs, simulator/model environment isolation, descendant
cleanup, cumulative save intervals and resumed warmup windows. These CPU collectives do not validate
NCCL or native distributed gradients. `test_native_distributed_ppo` uses two real GPUs to check
native initial broadcast, independent trajectory sampling and equal updated policy/critic/Adam;
it is skipped when fewer than two GPUs are visible. Real GR00T/Kit scaling requires separate runs.
Changing probability `sum` to `mean` or using reset state for timeout bootstrap makes the
corresponding test fail. No native PPO/GAE implementation is copied.

To additionally verify partial resets in a real two-environment PhysX scene with both RGB camera
views per robot, set `GR00T_SIM_TESTS=1` on the same pytest command. This opt-in test launches Kit
only in a simulator-environment child process and requires the task's USD assets. Run it when the
GPU is free; the parent remains in the model environment.
