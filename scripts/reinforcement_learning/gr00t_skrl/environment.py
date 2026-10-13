# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""skrl environment wrapper bridging full-chain actions to seven physical controls."""

from __future__ import annotations

import socket
import time

import gymnasium as gym
import numpy as np
import torch
from skrl.envs.wrappers.torch import Wrapper

from .policy import ChainPolicy, FrozenEncoder
from .protocol import Request, RpcConnection, RunConfig, Transition


class RemoteEnvironment(Wrapper):
    """Vector auto-reset environment with frozen features and state-only critic inputs."""

    def __init__(self, cfg: RunConfig, encoder: FrozenEncoder, policy: ChainPolicy):
        # The remote interface itself owns the skrl spaces; no local Gym environment is built.
        self._device = encoder.device
        self._observation_space = policy.observation_space
        self._action_space = policy.action_space
        self._state_space = gym.spaces.Box(-np.inf, np.inf, (8,), dtype=np.float32)
        self._num_envs = cfg.num_envs
        self.encoder = encoder
        self.policy = policy
        self._state = None
        self._observations = None
        self._autoreset_pending = False
        self._closed = False
        self.metrics: list[dict] = []
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(cfg.rpc_timeout)
        deadline = time.monotonic() + cfg.rpc_timeout
        try:
            while True:
                try:
                    connection.connect(cfg.socket_path)
                    break
                except (FileNotFoundError, ConnectionRefusedError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Simulation did not accept a model connection") from None
                    time.sleep(0.1)
        except BaseException:
            connection.close()
            raise
        self.rpc = RpcConnection(connection, cfg.rpc_timeout)

    @property
    def num_envs(self) -> int:
        """Number of independently resetting physical environments."""
        return self._num_envs

    @property
    def num_agents(self) -> int:
        """One native PPO agent controls the robot."""
        return 1

    @property
    def observation_space(self) -> gym.Space:
        """Fixed frozen-feature observation space."""
        return self._observation_space

    @property
    def state_space(self) -> gym.Space:
        """Eight-dimensional physical critic state."""
        return self._state_space

    @property
    def action_space(self) -> gym.Space:
        """Complete stochastic generation-chain space stored by native memory."""
        return self._action_space

    def reset(self) -> tuple[torch.Tensor, dict]:
        """Reuse the received auto-reset observation when native trainer requests reset."""
        if self._autoreset_pending:
            self._autoreset_pending = False
            return self._observations, {}
        reply = self._request(Request("reset"))
        return self._encode(reply), {}

    def state(self) -> torch.Tensor:
        """Return the normal, possibly reset-after-terminal critic state."""
        return self._state

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Decode only x_K's first action; preserve final critic state independently."""
        expected = (self.num_envs, self.policy.num_actions)
        if actions.shape != expected or not torch.isfinite(actions).all():
            raise ValueError(f"Expected finite full-chain actions of shape {expected}")
        chain = actions.reshape(
            self.num_envs, self.policy.generation_steps + 1, self.policy.horizon, self.policy.action_dim
        )
        command = self.encoder.decode(chain[:, -1])
        reply = self._request(Request("step", command))
        if not np.array_equal(reply.executed_action, command):
            raise RuntimeError("Simulator did not execute the requested seven-dimensional command")
        next_observations = self._encode(reply)
        infos = {}
        if reply.final_state is not None:
            infos["final_state"] = torch.as_tensor(reply.final_state, device=self.device)
        # skrl requests explicit reset only for a single environment; vector envs auto-reset in Isaac Lab.
        self._autoreset_pending = self.num_envs == 1 and bool((reply.terminated | reply.truncated).any())
        step = len(self.metrics) // self.num_envs + 1
        for index in range(self.num_envs):
            self.metrics.append(
                {
                    "env_id": index,
                    "environment_step": step,
                    "command": command[index].tolist(),
                    "motion_m": float(reply.motion[index]),
                    "reward": float(reply.reward[index]),
                    "terminated": bool(reply.terminated[index]),
                    "truncated": bool(reply.truncated[index]),
                    "table_rgb_std": float(reply.observation.table_rgb[index].std()),
                    "wrist_rgb_std": float(reply.observation.wrist_rgb[index].std()),
                }
            )
        tensors = [
            torch.tensor(value, device=self.device).reshape(self.num_envs, 1)
            for value in (reply.reward, reply.terminated, reply.truncated)
        ]
        return next_observations, *tensors, infos

    def render(self) -> None:
        """Cameras are rendered inside the simulation child."""

    def close(self) -> None:
        """Close gracefully when possible; always release the local socket."""
        if not self._closed:
            self._closed = True
            try:
                self.rpc.send(Request("close"))
            except OSError:
                pass
            finally:
                self.rpc.close()

    def _request(self, request: Request) -> Transition:
        request.validate(self.num_envs)
        self.rpc.send(request)
        reply = self.rpc.receive()
        if isinstance(reply, str):
            raise RuntimeError(f"Simulation failed: {reply}")
        if not isinstance(reply, Transition):
            raise ValueError("Invalid simulation reply")
        reply.validate(self.num_envs)
        return reply

    def _encode(self, reply: Transition) -> torch.Tensor:
        self._state = torch.as_tensor(reply.observation.state, device=self.device)
        self._observations = self.encoder.encode(reply.observation)
        return self._observations
