# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Local, bounded RPC messages shared by the isolated model and simulation processes."""

from __future__ import annotations

import io
import pickle
import socket
import struct
from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class RunConfig:
    """Configuration written by the launcher and consumed by both processes."""

    model_path: str
    backbone_path: str
    run_dir: str
    socket_path: str
    task: str = "IsaacContrib-Stack-Cube-Franka-IK-Rel-Visuomotor"
    seed: int = 42
    timesteps: int = 2
    rollouts: int = 2
    learning_epochs: int = 2
    generation_steps: int = 2
    sigma: float = 0.05
    learning_rate: float = 1e-6
    max_tokens: int = 1024
    episode_steps: int | None = None
    resume: str | None = None
    mode: Literal["train", "inference"] = "train"
    rpc_timeout: float = 300.0
    instruction: str = "Stack the red cube on the blue cube, then stack the green cube on the red cube."
    viser: bool = False
    viser_port: int = 8080
    num_envs: int = 1
    encoder_batch_size: int = 1
    sampling_batch_size: int = 1
    training_batch_size: int = 1

    @property
    def mini_batches(self) -> int:
        """Number of native minibatches, each bounded by training_batch_size."""
        samples = self.rollouts * self.num_envs
        return (samples + self.training_batch_size - 1) // self.training_batch_size

    def validate(self) -> None:
        """Reject invalid configurations before starting either process."""
        if min(self.timesteps, self.rollouts, self.learning_epochs, self.generation_steps, self.max_tokens) < 1:
            raise ValueError("Step counts and token capacity must be positive")
        sizes = (self.num_envs, self.encoder_batch_size, self.sampling_batch_size, self.training_batch_size)
        if any(type(size) is not int or size < 1 for size in sizes):
            raise ValueError("Environment count and batch sizes must be positive integers")
        if self.rollouts < 2 or self.timesteps % self.rollouts:
            raise ValueError("Use complete rollouts with at least two transitions for native GAE normalization")
        if not self.sigma > 0 or not np.isfinite(self.sigma):
            raise ValueError("sigma must be finite and positive")
        if not self.learning_rate > 0 or not np.isfinite(self.learning_rate):
            raise ValueError("learning_rate must be finite and positive")
        if self.rpc_timeout <= 0 or self.episode_steps is not None and self.episode_steps < 1:
            raise ValueError("Timeout and episode length must be positive")
        if not 1 <= self.viser_port <= 65535:
            raise ValueError("Viser port must be between 1 and 65535")
        if "nvidia/Cosmos-Reason2" not in self.backbone_path:
            raise ValueError("Keep the nvidia/Cosmos-Reason2 substring in the backbone path")


@dataclass
class Observation:
    """Two RGB images and XYZ [m], rotation vector [rad], finger positions [m]."""

    table_rgb: np.ndarray
    wrist_rgb: np.ndarray
    state: np.ndarray

    def validate(self, num_envs: int | None = None) -> None:
        """Check the wire contract before using an observation."""
        if self.state.ndim != 2 or self.state.shape[1] != 8 or self.state.shape[0] < 1:
            raise ValueError("Expected N x 8 state components")
        count = self.state.shape[0] if num_envs is None else num_envs
        if self.state.shape != (count, 8) or self.state.dtype != np.float32 or not np.isfinite(self.state).all():
            raise ValueError("Expected N x 8 finite float32 state components")
        for rgb in (self.table_rgb, self.wrist_rgb):
            if rgb.ndim != 4 or rgb.shape[0] != count or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
                raise ValueError("Expected N x H x W x 3 uint8 RGB images")
            if min(rgb.shape[1:3]) < 1:
                raise ValueError("RGB images must not be empty")


@dataclass
class Request:
    """Simulation command, with seven control values for a step."""

    command: Literal["reset", "step", "close"]
    action: np.ndarray | None = None

    def validate(self, num_envs: int | None = None) -> None:
        """Reject malformed commands and non-finite robot controls."""
        if self.command not in ("reset", "step", "close"):
            raise ValueError(f"Unknown command: {self.command}")
        if self.command == "step":
            if self.action is None or self.action.ndim != 2 or self.action.shape[1] != 7:
                raise ValueError("Expected N x 7 robot controls")
            count = self.action.shape[0] if num_envs is None else num_envs
            if (
                count < 1
                or self.action.shape != (count, 7)
                or self.action.dtype != np.float32
                or not np.isfinite(self.action).all()
            ):
                raise ValueError("Expected N x 7 finite float32 robot controls")
            if np.any(np.abs(self.action[:, :6]) > 0.100001) or not np.isin(self.action[:, 6], (-1, 1)).all():
                raise ValueError("Arm controls must be in [-0.1, 0.1] and gripper must be binary")


@dataclass
class Transition:
    """Reset-aware reply; final_state is captured before automatic reset."""

    observation: Observation
    reward: np.ndarray | None = None
    terminated: np.ndarray | None = None
    truncated: np.ndarray | None = None
    final_state: np.ndarray | None = None
    executed_action: np.ndarray | None = None
    motion: np.ndarray | None = None
    """End-effector displacement during the physics step [m], before any reset."""

    def __post_init__(self) -> None:
        count = self.observation.state.shape[0]
        if self.reward is None:
            self.reward = np.zeros(count, np.float32)
        if self.terminated is None:
            self.terminated = np.zeros(count, bool)
        if self.truncated is None:
            self.truncated = np.zeros(count, bool)
        if self.motion is None:
            self.motion = np.zeros(count, np.float32)

    def validate(self, num_envs: int) -> None:
        """Validate each environment's reply, including reset-before-terminal states."""
        self.observation.validate(num_envs)
        for value, dtype in (
            (self.reward, np.float32),
            (self.motion, np.float32),
            (self.terminated, np.bool_),
            (self.truncated, np.bool_),
        ):
            if (
                not isinstance(value, np.ndarray)
                or value.shape != (num_envs,)
                or value.dtype != dtype
                or not np.isfinite(value).all()
            ):
                raise ValueError("Invalid batched transition values")
        if self.final_state is not None:
            if (
                self.final_state.shape != (num_envs, 8)
                or self.final_state.dtype != np.float32
                or not np.isfinite(self.final_state).all()
            ):
                raise ValueError("Invalid batched final_state")
        elif (self.terminated | self.truncated).any():
            raise ValueError("Terminal transition requires final_state")
        if self.executed_action is not None:
            Request("step", self.executed_action).validate(num_envs)


class RpcConnection:
    """Length-framed RPC over a private Unix socket; all reads and writes have deadlines.

    Pickle is used only between launcher-owned local processes in a private directory.
    """

    def __init__(self, connection: socket.socket, timeout: float):
        self.connection = connection
        self.connection.settimeout(timeout)

    def send(self, message: Request | Transition | str | None) -> None:
        """Send one bounded message."""
        buffer = io.BytesIO()
        pickler = pickle.Pickler(buffer, protocol=5)
        pickler.dispatch_table = {np.ndarray: _reduce_array}
        pickler.dump(message)
        payload = buffer.getvalue()
        if len(payload) > 64 * 1024 * 1024:
            raise ValueError("RPC message exceeds 64 MiB")
        self.connection.sendall(struct.pack("!I", len(payload)) + payload)

    def receive(self) -> Request | Transition | str | None:
        """Receive a complete message, or raise on timeout/disconnection."""
        length = struct.unpack("!I", self._receive_exactly(4))[0]
        if length > 64 * 1024 * 1024:
            raise ValueError("RPC message exceeds 64 MiB")
        return pickle.loads(self._receive_exactly(length))

    def close(self) -> None:
        """Close the connection."""
        self.connection.close()

    def _receive_exactly(self, size: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < size:
            chunk = self.connection.recv(size - len(chunks))
            if not chunk:
                raise ConnectionError("Peer disconnected")
            chunks.extend(chunk)
        return bytes(chunks)


def _reduce_array(array: np.ndarray) -> tuple:
    # NumPy 2's default pickle refers to numpy._core, which is absent in the model's NumPy 1.26.
    if array.dtype.hasobject:
        raise ValueError("RPC does not accept object arrays")
    return _restore_array, (array.tobytes(), array.shape, array.dtype.str)


def _restore_array(data: bytes, shape: tuple[int, ...], dtype: str) -> np.ndarray:
    return np.frombuffer(data, dtype=np.dtype(dtype)).reshape(shape).copy()
