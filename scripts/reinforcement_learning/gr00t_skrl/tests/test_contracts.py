# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contracts unique to the GR00T adapter; no Kit is launched by the parent test process."""

from __future__ import annotations

import json
import math
import os
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
from skrl.agents.torch import ExperimentCfg
from skrl.agents.torch.ppo import PPO_CFG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch.sequential import SequentialTrainerCfg
from torch import nn
from transformers.feature_extraction_utils import BatchFeature

from scripts.reinforcement_learning.gr00t_skrl.agent import Gr00tPPO, RunState, validate_checkpoint
from scripts.reinforcement_learning.gr00t_skrl.environment import RemoteEnvironment
from scripts.reinforcement_learning.gr00t_skrl.policy import (
    ChainPolicy,
    FeatureLayout,
    FrozenEncoder,
    chain_log_probability,
    module_digest,
)
from scripts.reinforcement_learning.gr00t_skrl.protocol import (
    Observation,
    Request,
    RpcConnection,
    RunConfig,
    Transition,
)


class ScalarPolicy(GaussianMixin, Model):
    """Small genuine skrl Gaussian policy used for independent timeout return calculations."""

    def __init__(self):
        space = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        Model.__init__(self, observation_space=space, action_space=space, device="cpu")
        GaussianMixin.__init__(self, clip_actions=False)
        self.mean = nn.Parameter(torch.zeros(1))
        self.log_std = nn.Parameter(torch.zeros(1))

    def compute(self, inputs: dict, role: str = "") -> tuple:
        return self.mean.expand(inputs["observations"].shape[0], 1), {"log_std": self.log_std}


class ScalarValue(DeterministicMixin, Model):
    """V(s)=s for hand-calculated returns, with a trainable value scale."""

    def __init__(self):
        space = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        Model.__init__(self, observation_space=space, state_space=space, action_space=space, device="cpu")
        DeterministicMixin.__init__(self, clip_actions=False)
        self.scale = nn.Parameter(torch.ones(1))

    def compute(self, inputs: dict, role: str = "") -> tuple:
        return inputs["states"] * self.scale, {}


def make_agent(directory: Path, num_envs: int = 1, training_batch_size: int = 1) -> Gr00tPPO:
    """Construct native PPO with adapter lifecycle hooks on a small CPU fixture."""
    policy = ScalarPolicy()
    value = ScalarValue()
    cfg = PPO_CFG(
        rollouts=2,
        learning_epochs=2,
        mini_batches=math.ceil(2 * num_envs / training_batch_size),
        discount_factor=0.5,
        learning_rate=0.001,
        time_limit_bootstrap=True,
        experiment=ExperimentCfg(
            directory=str(directory), experiment_name="native", write_interval=0, checkpoint_interval=0
        ),
    )
    agent = Gr00tPPO(
        encoder=None,
        run_state=RunState(
            {"policy_version": "contract", "generation_steps": 2, "training_batch_size": training_batch_size}
        ),
        models={"policy": policy, "value": value},
        memory=RandomMemory(memory_size=2, num_envs=num_envs, device="cpu"),
        observation_space=policy.observation_space,
        state_space=value.state_space,
        action_space=policy.action_space,
        cfg=cfg,
        device="cpu",
    )
    agent.init(trainer_cfg=SequentialTrainerCfg(timesteps=2))
    agent.enable_training_mode(True)
    return agent


def fill_episode_boundaries(agent: Gr00tPPO) -> None:
    """One timeout then simultaneous true termination+timeout, with deliberately different reset states."""
    for index, (state, reset_state, terminal, reward, terminated) in enumerate(
        ((1.0, 100.0, 7.0, 10.0, False), (100.0, 200.0, 9.0, 20.0, True))
    ):
        state = torch.tensor([[state]])
        with torch.no_grad():
            action, _ = agent.act(torch.zeros(1, 1), state, timestep=index, timesteps=2)
        agent.record_transition(
            observations=torch.zeros(1, 1),
            states=state,
            actions=action,
            rewards=torch.tensor([[reward]]),
            next_observations=torch.zeros(1, 1),
            next_states=torch.tensor([[reset_state]]),
            terminated=torch.tensor([[terminated]]),
            truncated=torch.tensor([[True]]),
            infos={"final_state": torch.tensor([[terminal]])},
            timestep=index,
            timesteps=2,
        )


def test_full_chain_probability_and_native_memory() -> None:
    """Native memory retains all chain coordinates, and likelihood sums every Gaussian term."""
    chain = torch.tensor([[[[0.0, 1.0]], [[0.5, 2.0]], [[2.0, 3.0]]]])
    means = torch.tensor([[[[0.0, 1.0]], [[1.0, 2.0]]]], requires_grad=True)
    memory = RandomMemory(memory_size=1, num_envs=1, device="cpu")
    memory.create_tensor(name="actions", size=6, dtype=torch.float32)
    memory.add_samples(actions=chain.flatten(1))
    saved = next(iter(memory.sample(names=["actions"], batch_size=1, mini_batches=1)))[0].reshape(1, 3, 1, 2)
    # Four scalar residuals: 0.5, 1, 1, 1; normalizing constant counted four times.
    expected = -4 * math.log(0.5 * math.sqrt(2 * math.pi)) - (0.25 + 1 + 1 + 1) / (2 * 0.5**2)
    probability = chain_log_probability(saved, means, 0.5)
    assert probability.item() == pytest.approx(expected, abs=1e-6)
    probability.backward()
    torch.testing.assert_close(means.grad.flatten(), torch.tensor([2.0, 4.0, 4.0, 4.0]))
    # Including x_0 contributes exactly the same parameter-independent density to both policies.
    changed = means.detach() + 0.125
    old_density = probability.detach()
    new_density = chain_log_probability(saved, changed, 0.5)
    initial_constant = -math.log(2 * math.pi) - 0.5
    torch.testing.assert_close(
        new_density - old_density, (new_density + initial_constant) - (old_density + initial_constant)
    )
    with pytest.raises(ValueError, match="complete"):
        chain_log_probability(saved[:, 1:], means, 0.5)


@pytest.mark.parametrize("sampling_batch_size", [1, 2])
def test_original_head_recompute_and_trainable_features(sampling_batch_size: int) -> None:
    """A small original GR00T head retains gradients through projectors and has no dropout drift."""
    torch.manual_seed(4)
    config = Gr00tN1d7Config(
        hidden_size=8,
        input_embedding_dim=8,
        backbone_embedding_dim=8,
        max_state_dim=2,
        max_action_dim=2,
        action_horizon=2,
        max_num_embodiments=1,
        max_seq_len=8,
        use_vlln=True,
        vl_self_attention_cfg={"num_layers": 1, "num_attention_heads": 1, "attention_head_dim": 8, "dropout": 0.2},
        diffusion_model_cfg={
            "num_layers": 2,
            "num_attention_heads": 1,
            "attention_head_dim": 8,
            "output_dim": 8,
            "dropout": 0.2,
            "interleave_self_attention": True,
        },
    )
    head = Gr00tN1d7ActionHead(config)
    layout = FeatureLayout(4, 8, 2)
    output = BatchFeature(
        data={
            "backbone_features": torch.randn(2, 3, 8),
            "backbone_attention_mask": torch.ones(2, 3, dtype=torch.bool),
            "image_mask": torch.tensor([[True, False, False]]).expand(2, -1),
        }
    )
    observations = layout.pack(output, torch.randn(2, 1, 2), torch.zeros(2, dtype=torch.long))
    policy = ChainPolicy(head, layout, 2, 0.05, "cpu", sampling_batch_size=sampling_batch_size)
    policy.train(True)
    with torch.no_grad():
        chain, sampled = policy.act({"observations": observations})
    _, recomputed = policy.act({"observations": observations, "taken_actions": chain})
    assert chain.shape == (2, 12)
    torch.testing.assert_close(sampled["log_prob"], recomputed["log_prob"], atol=0.01, rtol=0)
    assert not any(module.training for module in policy.modules())
    recomputed["log_prob"].sum().backward()
    for module in (head.state_encoder, head.vlln, head.vl_self_attention, head.model, head.action_decoder):
        assert any(parameter.grad is not None and parameter.grad.abs().max() > 0 for parameter in module.parameters())
    with pytest.raises(ValueError, match="entire saved"):
        policy.act({"observations": observations, "taken_actions": chain[:, 4:]})


def test_terminal_bootstrap_and_new_process_resume(tmp_path: Path) -> None:
    """Native returns isolate resets, and a new process restores models, moments, counters and RNG."""
    torch.manual_seed(7)
    agent = make_agent(tmp_path)
    fill_episode_boundaries(agent)
    agent.post_interaction(timestep=0, timesteps=2)
    agent.post_interaction(timestep=1, timesteps=2)
    # Timeout: 10 + gamma*7. True termination: 20, even with truncated=True.
    torch.testing.assert_close(agent.memory.get_tensor_by_name("returns").flatten(), torch.tensor([13.5, 20.0]))
    assert agent.run_state.optimizer_steps == 4
    checkpoint_path = Path(agent.experiment_dir) / "checkpoints/agent_2.pt"
    before = module_digest(agent.policy)
    # Restore the saved RNG to compute what should follow the checkpoint, without relying on current state.
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    torch.set_rng_state(saved["run_state"]["rng"]["torch"])
    expected_rng = torch.rand(3).tolist()
    first_state = next(iter(saved["optimizer"]["state"].values()))
    expected_moment = first_state["exp_avg"].flatten().tolist()
    child = subprocess.run(
        [sys.executable, str(Path(__file__).absolute()), str(checkpoint_path), str(tmp_path / "resume")],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).absolute().parents[4]), "NO_ALBUMENTATIONS_UPDATE": "1"},
        check=False,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert child.returncode == 0, child.stderr
    result = json.loads(child.stdout.strip().splitlines()[-1])
    assert result["digest"] == before
    assert result["moment"] == expected_moment
    assert result["random"] == expected_rng
    assert result["optimizer_steps"] == 8
    assert result["environment_steps"] == 4
    with pytest.raises(ValueError, match="Incompatible"):
        validate_checkpoint(str(checkpoint_path), {"policy_version": "contract", "generation_steps": 3})
    with pytest.raises(ValueError, match="requires.*final_state"):
        agent.record_transition(
            observations=torch.zeros(1, 1),
            states=torch.zeros(1, 1),
            actions=torch.zeros(1, 1),
            rewards=torch.zeros(1, 1),
            next_observations=torch.zeros(1, 1),
            next_states=torch.zeros(1, 1),
            terminated=torch.tensor([[False]]),
            truncated=torch.tensor([[True]]),
            infos={},
            timestep=2,
            timesteps=4,
        )


def test_parallel_terminal_returns_and_counters(tmp_path: Path) -> None:
    """Timeout, true termination and a continuing episode cannot contaminate one another."""
    agent = make_agent(tmp_path, num_envs=3)
    for index, (states, next_states, rewards, terminated, truncated, final_states) in enumerate(
        (
            ([1, 2, 3], [100, 200, 4], [10, 20, 30], [False, True, False], [True, True, False], [7, 999, 4]),
            ([100, 200, 4], [8, 9, 5], [40, 50, 60], [True, False, False], [False, True, False], [8, 11, 5]),
        )
    ):
        states = torch.tensor(states, dtype=torch.float32).reshape(3, 1)
        with torch.no_grad():
            actions, _ = agent.act(torch.zeros(3, 1), states, timestep=index, timesteps=2)
        agent.record_transition(
            observations=torch.zeros(3, 1),
            states=states,
            actions=actions,
            rewards=torch.tensor(rewards, dtype=torch.float32).reshape(3, 1),
            next_observations=torch.zeros(3, 1),
            next_states=torch.tensor(next_states, dtype=torch.float32).reshape(3, 1),
            terminated=torch.tensor(terminated).reshape(3, 1),
            truncated=torch.tensor(truncated).reshape(3, 1),
            infos={"final_state": torch.tensor(final_states, dtype=torch.float32).reshape(3, 1)},
            timestep=index,
            timesteps=2,
        )
        agent.post_interaction(timestep=index, timesteps=2)
    # Continuing env: 30 + .5*4 + .5*.95*(60 + .5*5 - 4).
    expected = torch.tensor([[13.5, 20, 59.7875], [40, 55.5, 62.5]])
    torch.testing.assert_close(agent.memory.get_tensor_by_name("returns").squeeze(-1), expected)
    assert agent.run_state.environment_steps == 2
    assert agent.run_state.transitions == 6
    assert agent.run_state.optimizer_steps == 12


def test_legacy_checkpoint_resumes_with_parallel_memory(tmp_path: Path) -> None:
    """Legacy single-env moments and counters survive an environment-count change."""
    original = make_agent(tmp_path / "single")
    fill_episode_boundaries(original)
    original.post_interaction(timestep=0, timesteps=2)
    original.post_interaction(timestep=1, timesteps=2)
    path = tmp_path / "legacy.pt"
    original.save(str(path))
    saved = torch.load(path, map_location="cpu", weights_only=False)
    saved["run_state"].pop("transitions", None)
    saved["run_state"]["metadata"].pop("training_batch_size", None)
    torch.save(saved, path)
    parallel = make_agent(tmp_path / "parallel", num_envs=2)
    parallel.resume(str(path))
    assert module_digest(parallel.policy) == module_digest(original.policy)
    assert parallel.run_state.transitions == parallel.run_state.environment_steps == 2
    assert parallel.run_state.optimizer_steps == 4
    changed_batch = make_agent(tmp_path / "changed_batch", num_envs=2, training_batch_size=2)
    with pytest.raises(ValueError, match="Incompatible"):
        changed_batch.resume(str(path))


def test_parallel_wire_contract() -> None:
    """Batched messages reject inconsistent counts, malformed masks and unsafe controls."""
    rgb = np.zeros((2, 2, 3, 3), dtype=np.uint8)
    observation = Observation(rgb, rgb.copy(), np.zeros((2, 8), np.float32))
    observation.validate(2)
    command = np.zeros((2, 7), np.float32)
    command[:, 6] = [-1, 1]
    Request("step", command).validate(2)
    reply = Transition(
        observation,
        np.zeros(2, np.float32),
        np.zeros(2, bool),
        np.array([True, False]),
        np.zeros((2, 8), np.float32),
        command,
        np.zeros(2, np.float32),
    )
    reply.validate(2)
    with pytest.raises(ValueError):
        observation.validate(1)
    with pytest.raises(ValueError):
        Request("step", command[:1]).validate(2)
    command[1, 0] = 0.2
    with pytest.raises(ValueError, match="Arm controls"):
        Request("step", command).validate(2)
    reply.final_state = None
    with pytest.raises(ValueError, match="final_state"):
        reply.validate(2)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_encoder_batches_and_per_environment_decode(batch_size: int) -> None:
    """Processor receives each environment's own images and signed reference state, including the tail batch."""
    batches = []

    class Processor:
        def __call__(self, messages):
            step = messages[0]["content"]
            return {"state": np.concatenate(list(step.states.values()), axis=-1), "image": step.images["image"]}

        def collator(self, items):
            batches.append(len(items))
            return {
                "inputs": {
                    "state": torch.from_numpy(np.stack([item["state"] for item in items])),
                    "embodiment_id": torch.zeros(len(items), dtype=torch.long),
                    "images": torch.tensor([int(item["image"][0, 0, 0, 0]) for item in items]),
                }
            }

        def decode_action(self, action, embodiment, states):
            return {
                key: states[key][:, :, :1] + action[:, :, index : index + 1]
                for index, key in enumerate(("x", "y", "z", "roll", "pitch", "yaw", "gripper"))
            }

    def backbone(inputs):
        markers = inputs["images"].float().reshape(-1, 1, 1).expand(-1, 2, 2)
        return BatchFeature(
            data={
                "backbone_features": markers,
                "backbone_attention_mask": torch.ones(markers.shape[:2], dtype=torch.bool),
                "image_mask": torch.ones(markers.shape[:2], dtype=torch.bool),
            }
        )

    encoder = object.__new__(FrozenEncoder)
    encoder.processor = Processor()
    encoder.batch_size = batch_size
    encoder.embodiment = "libero_sim"
    encoder.instruction = "test"
    encoder.layout = FeatureLayout(4, 2, 8)
    encoder.model = SimpleNamespace(prepare_input=lambda inputs: (inputs, inputs))
    encoder.backbone = backbone
    rgb = np.zeros((3, 2, 2, 3), np.uint8)
    rgb[:, 0, 0, 0] = [11, 22, 33]
    state = np.zeros((3, 8), np.float32)
    state[:, 0] = [0.05, -0.05, 0.2]
    state[:, 6:] = [[0.01, -0.01], [-0.02, 0.02], [0.03, -0.03]]
    features = encoder.encode(Observation(rgb, rgb.copy(), state))
    assert batches == ([1, 1, 1] if batch_size == 1 else [2, 1])
    torch.testing.assert_close(features[:, 0], torch.tensor([11.0, 22.0, 33.0]))
    np.testing.assert_array_equal(encoder.last_states["gripper"][:, 0], state[:, 6:])
    commands = encoder.decode(torch.zeros(3, 2, 7))
    np.testing.assert_allclose(commands[:, 0], [0.05, -0.05, 0.1])
    np.testing.assert_array_equal(commands[:, 6], [1, -1, 1])


@pytest.mark.parametrize("num_envs", [1, 2])
def test_remote_autoreset_and_action_identity(tmp_path: Path, num_envs: int) -> None:
    """Single-env reset is consumed, and a partial vector reset leaves other environments running."""
    space = gym.spaces.Box(-np.inf, np.inf, (8,), np.float32)
    policy = SimpleNamespace(
        observation_space=space, action_space=space, num_actions=1, generation_steps=0, horizon=1, action_dim=1
    )

    class Encoder:
        device = torch.device("cpu")

        def encode(self, observation):
            self.state = observation.state.copy()
            return torch.tensor(self.state)

        def decode(self, action):
            commands = np.zeros((num_envs, 7), np.float32)
            commands[:, 0] = self.state[:, 0] / 1000
            commands[:, 6] = 1
            return commands

    cfg = RunConfig("model", "nvidia/Cosmos-Reason2-2B", str(tmp_path), str(tmp_path / "env.sock"), num_envs=num_envs)
    requests = []

    def reply(state):
        rgb = np.zeros((num_envs, 2, 2, 3), np.uint8)
        return Observation(rgb, rgb.copy(), state)

    def serve(server):
        connection, _ = server.accept()
        rpc = RpcConnection(connection, 5)
        try:
            initial = np.zeros((num_envs, 8), np.float32)
            initial[:, 0] = np.arange(1, num_envs + 1)
            requests.append(rpc.receive())
            rpc.send(Transition(reply(initial)))
            requests.append(rpc.receive())
            reset = initial.copy()
            reset[0, 0] = 100
            done = np.zeros(num_envs, bool)
            done[0] = True
            rpc.send(
                Transition(
                    reply(reset),
                    truncated=done,
                    final_state=initial + 7,
                    executed_action=requests[-1].action,
                    motion=np.ones(num_envs, np.float32),
                )
            )
            requests.append(rpc.receive())
            rpc.send(Transition(reply(reset), executed_action=requests[-1].action))
            requests.append(rpc.receive())
        finally:
            rpc.close()

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server, ThreadPoolExecutor(max_workers=1) as pool:
        server.bind(cfg.socket_path)
        server.listen(1)
        server.settimeout(5)
        worker = pool.submit(serve, server)
        env = RemoteEnvironment(cfg, Encoder(), policy)
        try:
            env.reset()
            observations, _, _, truncated, infos = env.step(torch.zeros(num_envs, 1))
            assert truncated[:, 0].tolist() == ([True] if num_envs == 1 else [True, False])
            assert infos["final_state"][0, 0] == 8
            if num_envs == 1:
                cached, _ = env.reset()
                torch.testing.assert_close(cached, observations)
            env.step(torch.zeros(num_envs, 1))
        finally:
            env.close()
        worker.result(timeout=5)
    assert [request.command for request in requests] == ["reset", "step", "step", "close"]
    np.testing.assert_allclose(requests[2].action[:, 0], [0.1] if num_envs == 1 else [0.1, 0.002])


def test_parallel_minibatches_cover_all_samples() -> None:
    """An uneven native split covers every environment/time pair exactly once."""
    memory = RandomMemory(memory_size=2, num_envs=3, device="cpu")
    memory.create_tensor(name="actions", size=1, dtype=torch.float32)
    memory.add_samples(actions=torch.tensor([[0.0], [1.0], [2.0]]))
    memory.add_samples(actions=torch.tensor([[3.0], [4.0], [5.0]]))
    batches = memory.sample(names=["actions"], batch_size=6, mini_batches=math.ceil(6 / 4))
    assert all(0 < batch[0].shape[0] <= 4 for batch in batches)
    torch.testing.assert_close(torch.cat([batch[0] for batch in batches]).flatten().sort().values, torch.arange(6.0))


@pytest.mark.skipif(os.environ.get("GR00T_SIM_TESTS") != "1", reason="Opt-in real PhysX/two-camera test")
def test_native_parallel_partial_reset(tmp_path: Path) -> None:
    """Run Kit only in a child and verify a timeout cannot reset another physical robot."""
    repo = Path(__file__).absolute().parents[4]
    script = tmp_path / "partial_reset.py"
    script.write_text("""
import gymnasium as gym
import numpy as np
import torch
from scripts.reinforcement_learning.gr00t_skrl.simulation import (
    LocalEvents, LocalRewards, observation_packet, parse_env_cfg, launch_simulation,
)
task = "IsaacContrib-Stack-Cube-Franka-IK-Rel-Visuomotor"
cfg = parse_env_cfg(task, device="cuda:0", num_envs=2, overrides=["physics=isaacsim_physx"])
cfg.scene.plane.spawn.visible = True
cfg.events = LocalEvents()
cfg.rewards = LocalRewards()
cfg.compute_final_obs = True
for camera in (cfg.scene.table_cam, cfg.scene.wrist_cam):
    camera.data_types = ["rgb"]
with launch_simulation(cfg, {"headless": True, "enable_cameras": True}):
    env = gym.make(task, cfg=cfg).unwrapped
    try:
        observations, _ = env.reset(seed=42)
        initial = observation_packet(observations)
        initial.validate(2)
        # Cloned origins differ in world space; model XYZ must be local and equal initially.
        np.testing.assert_allclose(initial.state[0, :3], initial.state[1, :3], atol=1e-5)
        env.episode_length_buf[0] = env.max_episode_length - 1
        controls = torch.zeros((2, 7), device=env.device)
        controls[:, 6] = 1
        controls[1, 0] = .02
        observations, reward, terminated, truncated, infos = env.step(controls)
        assert truncated.tolist() == [True, False], truncated
        assert not terminated.any(), terminated
        assert env.episode_length_buf.tolist() == [0, 1], env.episode_length_buf
        packet = observation_packet(observations)
        packet.validate(2)
        terminal = observation_packet(infos["final_obs"])
        terminal.validate(2)
        np.testing.assert_allclose(packet.state[0, :3], initial.state[0, :3], atol=1e-5)
        assert np.linalg.norm(packet.state[1, :3] - initial.state[1, :3]) > 1e-6
        for camera in (packet.table_rgb, packet.wrist_rgb):
            assert all(image.std() > 1 for image in camera)
        observations, _, _, truncated, _ = env.step(controls)
        assert not truncated.any(), truncated
        assert env.episode_length_buf.tolist() == [1, 2], env.episode_length_buf
        print("Partial reset preserved the other environment, local state and both cameras")
    finally:
        env.close()
""")
    child = subprocess.run(
        ["uv", "run", "--project", str(repo), "--no-sync", "python", str(script)],
        cwd=repo,
        env={**os.environ, "PYTHONPATH": str(repo), "OMNI_KIT_ACCEPT_EULA": "YES"},
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert child.returncode == 0, child.stdout[-6000:] + child.stderr[-6000:]


@pytest.mark.parametrize("field", ["num_envs", "encoder_batch_size", "sampling_batch_size", "training_batch_size"])
def test_parallel_config_requires_positive_sizes(field: str) -> None:
    cfg = RunConfig("model", "nvidia/Cosmos-Reason2-2B", "run", "socket", **{field: 0})
    with pytest.raises(ValueError):
        cfg.validate()


def test_numpy_versions_and_rpc_deadline(tmp_path: Path) -> None:
    """Real NumPy 2 simulator-side arrays arrive intact in NumPy 1 model-side RPC, and EOF cannot hang."""
    repo = Path(__file__).absolute().parents[4]
    socket_path = str(tmp_path / "transport.sock")
    script = """
import socket
import numpy as np
from scripts.reinforcement_learning.gr00t_skrl.protocol import Observation, RpcConnection, Transition
connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
connection.connect(SOCKET_PATH)
rpc = RpcConnection(connection, 15)
assert np.__version__.startswith("2.")
rgb = np.arange(36, dtype=np.uint8).reshape(2, 2, 3, 3)
state = np.arange(16, dtype=np.float32).reshape(2, 8)
reply = Transition(Observation(rgb, rgb.copy(), state), truncated=np.array([True, False]), final_state=state + 7)
reply.validate(2)
rpc.send(reply)
request = rpc.receive()
request.validate()
expected = np.array([[0.1, -0.1, 0, 0, 0, 0, 1], [-0.1, 0, 0, 0, 0, 0, -1]], np.float32)
np.testing.assert_array_equal(request.action, expected)
rpc.close()
""".replace("SOCKET_PATH", repr(socket_path))
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(socket_path)
        server.listen(1)
        server.settimeout(15)
        child = subprocess.Popen(
            ["uv", "run", "--project", str(repo), "--no-sync", "python", "-c", script],
            cwd=repo,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            connection, _ = server.accept()
            rpc = RpcConnection(connection, 15)
            reply = rpc.receive()
            assert isinstance(reply, Transition)
            reply.validate(2)
            np.testing.assert_array_equal(
                reply.observation.table_rgb, np.arange(36, dtype=np.uint8).reshape(2, 2, 3, 3)
            )
            np.testing.assert_array_equal(reply.final_state, np.arange(16, dtype=np.float32).reshape(2, 8) + 7)
            rpc.send(Request("step", np.array([[0.1, -0.1, 0, 0, 0, 0, 1], [-0.1, 0, 0, 0, 0, 0, -1]], np.float32)))
            _, stderr = child.communicate(timeout=15)
            assert child.returncode == 0, stderr
            with pytest.raises(ConnectionError, match="disconnected"):
                rpc.receive()
            rpc.close()
        finally:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=15)
    receiver, sender = socket.socketpair()
    with receiver, sender:
        rpc = RpcConnection(receiver, 0.05)
        with pytest.raises(TimeoutError):
            rpc.receive()


if __name__ == "__main__":
    checkpoint_path, output_dir = sys.argv[1:]
    agent = make_agent(Path(output_dir))
    agent.resume(checkpoint_path)
    digest = module_digest(agent.policy)
    moment = next(iter(agent.optimizer.state.values()))["exp_avg"].flatten().tolist()
    agent.run_state.restore_rng()
    restored_random = torch.rand(3).tolist()
    fill_episode_boundaries(agent)
    agent.post_interaction(timestep=0, timesteps=2)
    agent.post_interaction(timestep=1, timesteps=2)
    print(
        json.dumps(
            {
                "digest": digest,
                "moment": moment,
                "random": restored_random,
                "optimizer_steps": agent.run_state.optimizer_steps,
                "environment_steps": agent.run_state.environment_steps,
            }
        )
    )
