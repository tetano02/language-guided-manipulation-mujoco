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
    primitive_items = "\n".join(f"- {idx}: {name}" for idx, name in enumerate(primitives))
    schema_json = json.dumps(TASK_GRAPH_SCHEMA, indent=2, ensure_ascii=True)
    return f"""
You are a robotics planner. Convert primitive sequence into a compact high-level task graph.

Constraints:
- Return STRICT JSON only, no prose.
- Keep nodes executable by an EE-first controller.
- Parameters must stay dynamic by referencing environment state keys such as:
  - "target_ref": "env.target_pos"
  - "goal_ref": "env.goal_pos"
- Preserve temporal order from primitives.

Primitive sequence:
{primitive_items}

Required JSON schema:
{schema_json}
""".strip()

