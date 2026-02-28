"""Prompt helpers for task-graph generation."""

from __future__ import annotations

import json
from typing import Any, Sequence


TASK_GRAPH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["nodes", "edges"],
    "properties": {
        "nodes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "type", "params"],
                "properties": {
                    "id": {"type": "string"},
                    "type": {"type": "string"},
                    "params": {"type": "object"},
                },
            },
        },
        "edges": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["from", "to"],
                "properties": {
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                },
            },
        },
        "metadata": {"type": "object"},
    },
}


def build_task_graph_prompt(primitives: Sequence[str]) -> str:
    primitive_items = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(primitives))
    schema_json = json.dumps(TASK_GRAPH_SCHEMA, indent=2, ensure_ascii=True)
    return (
        "You are a robotics task planner for a Franka Panda arm performing pick-and-place.\n"
        "\n"
        "Your job: convert a noisy primitive sequence (from automatic segmentation) into a\n"
        "clean, correct task graph.\n"
        "\n"
        "IMPORTANT: the input primitives may be noisy, duplicated, or in wrong order because\n"
        "they come from automatic trajectory segmentation. You must:\n"
        "1. Understand the INTENT (pick-and-place) and produce the CORRECT canonical sequence.\n"
        "2. Remove duplicates and spurious primitives (e.g. extra stabilize or lower at the end).\n"
        "3. Ensure the correct temporal ordering.\n"
        "\n"
        "ALLOWED NODE TYPES (use ONLY these as the 'type' field):\n"
        '- "approach_target": move EE above the target cube. params: {"target_ref": "env.target_pos"}\n'
        '- "lower_to_grasp": descend to grasp height. params: {"target_ref": "env.target_pos"}\n'
        '- "close_gripper": close gripper to grasp. params: {"target_ref": "env.target_pos"}\n'
        '- "lift_target": lift the grasped cube upward. params: {"target_ref": "env.target_pos"}\n'
        '- "move_to_goal": transport cube to goal region. params: {"target_ref": "env.target_pos", "goal_ref": "env.goal_pos"}\n'
        '- "open_gripper": lower to place height and release. params: {"goal_ref": "env.goal_pos"}\n'
        '- "stabilize": hold position briefly. params: {}\n'
        "\n"
        "CANONICAL PICK-AND-PLACE SEQUENCE (the correct order is):\n"
        "  approach_target -> lower_to_grasp -> close_gripper -> lift_target -> move_to_goal -> open_gripper\n"
        "\n"
        "Rules:\n"
        '- You MUST include "open_gripper" as the final action node to release the cube.\n'
        '- "lift_target" MUST come BEFORE "move_to_goal" (lift before transport).\n'
        '- Parameters must use dynamic references ("env.target_pos", "env.goal_pos"), never hardcoded coords.\n'
        "- Return STRICT JSON only, no prose, no markdown fences.\n"
        "- Do NOT invent new node types.\n"
        "\n"
        "Input primitive sequence (noisy):\n"
        f"{primitive_items}\n"
        "\n"
        "Required JSON schema:\n"
        f"{schema_json}"
    )
