from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")
from mujoco_env.env import EnvConfig, PandaPickPlaceEnv


def test_env_reset_and_step_smoke(tmp_path) -> None:
    env = PandaPickPlaceEnv(
        EnvConfig(
            render=False,
            output_dir=tmp_path,
            frame_skip=2,
            max_steps=20,
        )
    )
    try:
        obs = env.reset(seed=7)
        expected_keys = {
            "qpos",
            "qvel",
            "ee_pos",
            "target_pos",
            "distractor_pos",
            "goal_pos",
            "gripper_state",
            "action",
            "contacts_summary",
        }
        assert expected_keys.issubset(obs.keys())
        action = np.array([0.52, 0.0, 0.20, 0.0], dtype=float)
        obs, reward, done, info = env.step(action)
        assert np.isfinite(obs["qpos"]).all()
        assert np.isfinite(obs["qvel"]).all()
        assert isinstance(obs["contacts_summary"]["ncon"], int)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "sanity_ok" in info
    finally:
        env.close()


def test_env_reset_deterministic_positions(tmp_path) -> None:
    env = PandaPickPlaceEnv(
        EnvConfig(
            render=False,
            output_dir=tmp_path,
            frame_skip=1,
            max_steps=2,
        )
    )
    try:
        first = env.reset(seed=123)
        second = env.reset(seed=123)
        np.testing.assert_allclose(first["target_pos"], second["target_pos"], atol=1e-9)
        np.testing.assert_allclose(first["distractor_pos"], second["distractor_pos"], atol=1e-9)
        np.testing.assert_allclose(first["goal_pos"], second["goal_pos"], atol=1e-9)
    finally:
        env.close()
