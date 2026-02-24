from __future__ import annotations

from llm_abstraction.generate_task_graph import generate_task_graph, validate_task_graph_json


def test_task_graph_schema_validation_valid() -> None:
    payload = {
        "nodes": [
            {"id": "n0", "type": "approach_target", "params": {"target_ref": "env.target_pos"}},
            {"id": "n1", "type": "move_to_goal", "params": {"target_ref": "env.target_pos", "goal_ref": "env.goal_pos"}},
        ],
        "edges": [{"from": "n0", "to": "n1"}],
    }
    ok, errors, normalized = validate_task_graph_json(payload)
    assert ok
    assert not errors
    assert normalized is not None


def test_task_graph_schema_validation_invalid() -> None:
    payload = {
        "nodes": [{"id": "n0", "type": "approach_target", "params": {}}],
        "edges": [{"from": "missing", "to": "n0"}],
    }
    ok, errors, _ = validate_task_graph_json(payload)
    assert not ok
    assert errors


def test_task_graph_generation_fallback_deterministic() -> None:
    primitives = ["approach", "lower", "close_gripper", "lift", "move_to_goal", "open_gripper"]
    graph_a = generate_task_graph(primitives, api_key=None, allow_stub_fallback=True)
    graph_b = generate_task_graph(primitives, api_key=None, allow_stub_fallback=True)
    assert graph_a == graph_b
    ok, errors, normalized = validate_task_graph_json(graph_a)
    assert ok, errors
    assert normalized is not None
    assert normalized["metadata"]["generator"] == "deterministic_stub"

