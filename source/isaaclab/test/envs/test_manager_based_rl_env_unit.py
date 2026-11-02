# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for manager-based RL environments."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import pytest
import torch

from isaaclab.envs import ManagerBasedRLEnv

pytestmark = pytest.mark.unit


def _make_env_with_policy_obs_terms(
    num_envs: int,
    terms: list[tuple[str, tuple[int, ...], tuple[float, float] | None]],
) -> ManagerBasedRLEnv:
    """Build an uninitialized env whose observation manager stubs drive space setup.

    Args:
        num_envs: Number of vectorized environments.
        terms: Non-concatenated policy observation terms as ``(name, shape, clip)``
            where ``clip`` is ``(low, high)`` or ``None``.

    Returns:
        An uninitialized :class:`~isaaclab.envs.ManagerBasedRLEnv` ready for
        :meth:`~isaaclab.envs.ManagerBasedRLEnv._configure_gym_env_spaces`.
    """
    term_names = [name for name, _, _ in terms]
    term_dims = [shape for _, shape, _ in terms]
    term_cfgs = [SimpleNamespace(clip=clip) for _, _, clip in terms]

    env = object.__new__(ManagerBasedRLEnv)
    env._is_closed = True
    env.scene = SimpleNamespace(num_envs=num_envs)
    env.observation_manager = SimpleNamespace(
        active_terms={"policy": term_names},
        group_obs_concatenate={"policy": False},
        group_obs_dim={"policy": term_dims},
        _group_obs_term_cfgs={"policy": term_cfgs},
    )
    env.action_manager = SimpleNamespace(action_term_dim=[0])
    return env


def test_non_concatenated_obs_groups_contain_all_terms():
    """Non-concatenated observation groups expose every term in the Dict space (issue #3133).

    Before the fix, only the last term in each non-concatenated group would be present
    in the observation space Dict. This test ensures all terms are correctly included.
    """
    terms = [
        ("image", (4, 5, 3), (0.0, 1.0)),
        ("matrix", (2, 3), (-2.0, 2.0)),
        ("vector", (3,), None),
    ]
    env = _make_env_with_policy_obs_terms(num_envs=2, terms=terms)
    ManagerBasedRLEnv._configure_gym_env_spaces(env)

    assert isinstance(env.observation_space, gym.spaces.Dict)
    policy_space = env.observation_space.spaces["policy"]
    assert isinstance(policy_space, gym.spaces.Dict)

    expected_policy_terms = ["image", "matrix", "vector"]
    assert list(policy_space.spaces) == expected_policy_terms
    for term_name in expected_policy_terms:
        assert isinstance(policy_space.spaces[term_name], gym.spaces.Box)


def test_obs_space_follows_clip_constraint():
    """Observation space bounds reflect the clip constraint on each non-concatenated term."""
    terms = [
        ("vector", (3,), None),
        ("matrix", (2, 3), (-2.0, 2.0)),
        ("image", (4, 5, 3), (0.0, 1.0)),
    ]
    expected_shapes = {
        "vector": (2, 3),
        "matrix": (2, 2, 3),
        "image": (2, 4, 5, 3),
    }
    term_clips = {name: clip for name, _, clip in terms}
    env = _make_env_with_policy_obs_terms(num_envs=2, terms=terms)
    ManagerBasedRLEnv._configure_gym_env_spaces(env)

    for group_name, group_space in env.observation_space.spaces.items():
        assert isinstance(group_space, gym.spaces.Dict)
        for term_name, term_space in group_space.spaces.items():
            clip = term_clips[term_name]
            low = -np.inf if clip is None else clip[0]
            high = np.inf if clip is None else clip[1]
            assert isinstance(term_space, gym.spaces.Box), (
                f"Expected Box space for {term_name} in {group_name}, got {type(term_space)}"
            )
            assert term_space.shape == expected_shapes[term_name]
            assert np.all(term_space.low == low)
            assert np.all(term_space.high == high)


@pytest.mark.parametrize("auto_reset,manual_reset", [(True, False), (False, False), (False, True)])
def test_step_resets_only_when_owned_by_environment(auto_reset, manual_reset):
    """Caller-owned resets preserve done signals and terminal observations without reset side effects."""
    env = object.__new__(ManagerBasedRLEnv)
    env._is_closed = True
    env.cfg = SimpleNamespace(
        auto_reset=auto_reset,
        decimation=1,
        sim=SimpleNamespace(dt=0.02, render_interval=1),
        compute_final_obs=True,
        num_rerenders_on_reset=2,
    )
    env.scene = Mock(num_envs=2)
    env.sim = Mock(device="cpu", is_rendering=True)
    env.sim.consume_reset_request.return_value = manual_reset
    env.render_enabled = env.has_rtx_sensors = True
    env._physics_handles_decimation = False
    env._sim_step_counter = env.common_step_counter = 0
    env.episode_length_buf = torch.zeros(2, dtype=torch.long)
    env.action_manager = Mock()
    env.recorder_manager = Mock(active_terms=[])
    env.reward_manager = Mock()
    env.reward_manager.compute.return_value = torch.tensor([1.0, 0.0])
    env.termination_manager = Mock()
    env.termination_manager.terminated = torch.tensor([True, False])
    env.termination_manager.time_outs = torch.tensor([False, False])
    env.termination_manager.compute.return_value = torch.tensor([True, False])
    env.command_manager = Mock()
    env.event_manager = Mock(active_terms={})
    env.video_recorders = []
    env.extras = {}
    state = torch.tensor([3.0, 5.0])
    env.observation_manager = Mock()
    env.observation_manager.compute.side_effect = lambda **_: {"policy": state.clone()}
    env._reset_idx = Mock(side_effect=lambda ids: state.index_fill_(0, ids.long(), 0))

    observations, rewards, terminated, truncated, extras = env.step(torch.zeros(2, 0))

    assert rewards.tolist() == [1.0, 0.0]
    assert terminated.tolist() == [True, manual_reset]
    assert not truncated.any()
    assert observations["policy"].tolist() == ([0.0, 5.0] if auto_reset else [3.0, 5.0])
    assert env.recorder_manager.record_pre_reset.call_count == int(auto_reset)
    assert env.recorder_manager.record_post_reset.call_count == int(auto_reset)
    assert env._reset_idx.call_count == int(auto_reset)
    assert env.sim.render.call_count == (3 if auto_reset else 1)
    if auto_reset:
        assert extras["final_obs"]["policy"].tolist() == [3.0, 5.0]
    else:
        assert "final_obs" not in extras
