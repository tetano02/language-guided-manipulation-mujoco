"""Generate validated task graphs from primitive sequences using Gemini with retry + fallback."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from llm_abstraction.prompt import build_task_graph_prompt


ALLOWED_NODE_TYPES: tuple[str, ...] = (
    "approach_target",
    "lower_to_grasp",
    "close_gripper",
    "lift_target",
    "move_to_goal",
    "open_gripper",
    "stabilize",
)


class TaskNode(BaseModel):
    id: str
    type: Literal[
        "approach_target",
        "lower_to_grasp",
        "close_gripper",
        "lift_target",
        "move_to_goal",
        "open_gripper",
        "stabilize",
    ]
    params: dict[str, Any] = Field(default_factory=dict)


class TaskEdge(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    from_node: str = Field(alias="from")
    to_node: str = Field(alias="to")


class TaskGraph(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    nodes: list[TaskNode]
    edges: list[TaskEdge]
    metadata: dict[str, Any] = Field(default_factory=dict)


def validate_task_graph_json(payload: Any) -> tuple[bool, list[str], dict[str, Any] | None]:
    errors: list[str] = []
    try:
        graph = TaskGraph.model_validate(payload)
    except ValidationError as exc:
        return False, [str(exc)], None

    node_ids = {node.id for node in graph.nodes}
    if len(node_ids) != len(graph.nodes):
        errors.append("Duplicate node IDs detected.")
    for edge in graph.edges:
        if edge.from_node not in node_ids:
            errors.append(f"Edge source '{edge.from_node}' not found in nodes.")
        if edge.to_node not in node_ids:
            errors.append(f"Edge target '{edge.to_node}' not found in nodes.")
    if not graph.nodes:
        errors.append("Task graph must contain at least one node.")
    if errors:
        return False, errors, None
    return True, [], graph.model_dump(by_alias=True)


def _extract_json(text: str) -> dict[str, Any]:
    fenced = re.findall(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates = fenced + [text]
    for candidate in candidates:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end < 0 or end <= start:
            continue
        snippet = candidate[start : end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            continue
    raise ValueError("No valid JSON object found in LLM response.")


def _primitive_to_node_type(primitive: str) -> str:
    mapping = {
        "approach": "approach_target",
        "lower": "lower_to_grasp",
        "close_gripper": "close_gripper",
        "lift": "lift_target",
        "move_to_goal": "move_to_goal",
        "open_gripper": "open_gripper",
        "stabilize": "stabilize",
    }
    return mapping.get(primitive, "stabilize")


def _build_stub_graph(primitives: Iterable[str], *, reason: str) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    previous_node_id: str | None = None
    for idx, primitive in enumerate(primitives):
        node_id = f"n{idx}"
        node_type = _primitive_to_node_type(primitive)
        params: dict[str, Any] = {}
        if node_type in {"approach_target", "lower_to_grasp", "close_gripper", "lift_target"}:
            params["target_ref"] = "env.target_pos"
        if node_type in {"move_to_goal", "open_gripper"}:
            params["target_ref"] = "env.target_pos"
            params["goal_ref"] = "env.goal_pos"
        nodes.append({"id": node_id, "type": node_type, "params": params})
        if previous_node_id is not None:
            edges.append({"from": previous_node_id, "to": node_id})
        previous_node_id = node_id

    if not nodes:
        nodes = [{"id": "n0", "type": "stabilize", "params": {}}]
    graph = {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "generator": "deterministic_stub",
            "reason": reason,
            "allowed_node_types": list(ALLOWED_NODE_TYPES),
        },
    }
    ok, errors, normalized = validate_task_graph_json(graph)
    if not ok or normalized is None:
        raise ValueError(f"Internal stub validation failed: {errors}")
    return normalized


def _call_gemini(prompt: str, *, api_key: str, model_name: str) -> str:
    from google import genai

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model_name, contents=prompt)
    text = getattr(response, "text", None)
    if text:
        return str(text)
    raise ValueError("Gemini response did not contain text.")


def generate_task_graph(
    primitives: list[str],
    *,
    api_key: str | None,
    model_name: str = "gemini-2.0-flash",
    max_retries: int = 3,
    allow_stub_fallback: bool = True,
) -> dict[str, Any]:
    if not primitives:
        primitives = ["stabilize"]
    if not api_key:
        if allow_stub_fallback:
            return _build_stub_graph(primitives, reason="missing_api_key")
        raise ValueError("Gemini API key missing and stub fallback disabled.")

    prompt = build_task_graph_prompt(primitives)
    errors: list[str] = []
    for _ in range(max_retries):
        try:
            raw = _call_gemini(prompt, api_key=api_key, model_name=model_name)
            parsed = _extract_json(raw)
            ok, validation_errors, normalized = validate_task_graph_json(parsed)
            if ok and normalized is not None:
                normalized.setdefault("metadata", {})
                normalized["metadata"]["generator"] = "gemini"
                normalized["metadata"]["model_name"] = model_name
                return normalized
            errors.extend(validation_errors)
        except Exception as exc:
            errors.append(str(exc))

    if allow_stub_fallback:
        return _build_stub_graph(primitives, reason=f"llm_failed:{'; '.join(errors[:3])}")
    raise ValueError(f"Failed to generate valid task graph after retries: {errors}")


def _load_primitives(args: argparse.Namespace) -> list[str]:
    if args.primitives_json is not None:
        with args.primitives_json.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, list):
            raise ValueError("primitives_json must contain a JSON list of primitive names.")
        return [str(item) for item in payload]

    if args.segments_json is not None:
        with args.segments_json.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list) and payload and isinstance(payload[0], list):
            payload = payload[0]
        if not isinstance(payload, list):
            raise ValueError("segments_json must contain a segment list or list-of-lists.")
        return [str(segment["primitive"]) for segment in payload]

    raise ValueError("Either --primitives-json or --segments-json must be provided.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate validated task graph JSON.")
    parser.add_argument("--primitives-json", type=Path, default=None)
    parser.add_argument("--segments-json", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--api-key-env", type=str, default="GEMINI_API_KEY")
    parser.add_argument("--model-name", type=str, default="gemini-2.0-flash")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--no-stub-fallback", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    primitives = _load_primitives(args)
    api_key = os.getenv(args.api_key_env, "")
    graph = generate_task_graph(
        primitives,
        api_key=api_key if api_key else None,
        model_name=args.model_name,
        max_retries=args.max_retries,
        allow_stub_fallback=not args.no_stub_fallback,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(graph, handle, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()

