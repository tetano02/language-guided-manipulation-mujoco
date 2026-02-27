"""Smoke tests for the execution engine."""

from __future__ import annotations

import pytest

mujoco = pytest.importorskip("mujoco")

from execution.execute_from_graph import (  # noqa: E402
    NodeConfig,
    _build_execution_order,
    run_episode_from_graph,
)
from llm_abstraction.generate_task_graph import _build_stub_graph  # noqa: E402
from mujoco_env.env import EnvConfig, PandaPickPlaceEnv  # noqa: E402
from utils.logging import StructuredLogger  # noqa: E402


def _make_stub_graph() -> dict:
    """Build a deterministic stub task graph for testing."""
    primitives = ["approach", "lower", "close_gripper", "lift", "move_to_goal", "open_gripper"]
    return _build_stub_graph(primitives, reason="test")


def test_build_execution_order() -> None:
    """Execution order should be linear following edges."""
    graph = _make_stub_graph()
    order = _build_execution_order(graph)
    assert len(order) == len(graph["nodes"])
    assert order[0]["type"] == "approach_target"
    assert order[-1]["type"] == "open_gripper"


def test_execution_does_not_crash() -> None:
    """Run a few steps with the execution engine — no exceptions."""
    graph = _make_stub_graph()
    config = EnvConfig(seed=0, render=False, max_steps=30, sanity_asserts=False, log_jsonl=False)
    logger = StructuredLogger(enabled=False)
    node_cfg = NodeConfig(
        timeout_approach=5, timeout_lower=5, timeout_close=5,
        timeout_lift=5, timeout_move=5, timeout_open=5, timeout_stabilize=3,
    )
    with PandaPickPlaceEnv(config) as env:
        result = run_episode_from_graph(env, graph, logger, max_steps=30, node_config=node_cfg)
    assert "success" in result
    assert "steps_used" in result
    assert "grasp_achieved" in result
    assert "final_target_goal_dist" in result
    assert isinstance(result["steps_used"], int)
    assert result["steps_used"] > 0


def test_execution_result_keys() -> None:
    """Verify all expected keys are present in the result dict."""
    graph = _make_stub_graph()
    config = EnvConfig(seed=42, render=False, max_steps=20, sanity_asserts=False, log_jsonl=False)
    logger = StructuredLogger(enabled=False)
    node_cfg = NodeConfig(
        timeout_approach=3, timeout_lower=3, timeout_close=3,
        timeout_lift=3, timeout_move=3, timeout_open=3, timeout_stabilize=2,
    )
    with PandaPickPlaceEnv(config) as env:
        result = run_episode_from_graph(env, graph, logger, max_steps=20, node_config=node_cfg)
    expected_keys = {"success", "steps_used", "grasp_achieved", "grasp_retries",
                     "final_target_goal_dist", "final_target_z", "nodes_executed", "nodes_total"}
    assert expected_keys.issubset(result.keys())
