"""Execution engine: interpret a task graph and drive PandaPickPlaceEnv with dynamic sub-goals."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mujoco_env.env import EnvConfig, PandaPickPlaceEnv
from utils.logging import StructuredLogger


# ---------------------------------------------------------------------------
# Node-handler configuration (mirrors tuned scripted trajectory constants)
# ---------------------------------------------------------------------------

@dataclass
class NodeConfig:
    """Thresholds and heights used by node handlers, matching the tuned controller."""
    # Approach / lower
    pre_grasp_z_offset: float = 0.16
    pre_grasp_z_min: float = 0.18
    grasp_z_offset: float = 0.035
    grasp_z_min: float = 0.055
    approach_dist_threshold: float = 0.045
    descend_dist_threshold: float = 0.035
    # Lift
    lift_height_z: float = 0.28
    ee_lift_clearance_z: float = 0.06
    target_lift_clearance_z: float = 0.04
    floor_clear_required: int = 6
    # Move to goal
    target_goal_xy_threshold: float = 0.08
    # Place / open
    place_z: float = 0.08
    lower_to_open_z: float = 0.12
    open_hold_required: int = 18
    # Stabilize
    stabilize_steps: int = 10
    # Timeouts per node type
    timeout_approach: int = 80
    timeout_lower: int = 60
    timeout_close: int = 25
    timeout_lift: int = 100
    timeout_move: int = 100
    timeout_open: int = 80
    timeout_stabilize: int = 10
    # Grasp retry
    max_grasp_retries: int = 3


# ---------------------------------------------------------------------------
# Node handler results
# ---------------------------------------------------------------------------

@dataclass
class NodeResult:
    """Result of executing a single task-graph node."""
    completed: bool
    timed_out: bool
    steps_used: int
    aborted: bool = False
    retry_approach: bool = False  # request to re-run from approach


# ---------------------------------------------------------------------------
# Individual node handlers
# ---------------------------------------------------------------------------

def _run_approach(
    env: PandaPickPlaceEnv,
    obs: dict[str, Any],
    cfg: NodeConfig,
    logger: StructuredLogger,
) -> tuple[dict[str, Any], NodeResult]:
    """Move EE above the target cube."""
    for step in range(cfg.timeout_approach):
        target = np.asarray(obs["target_pos"], dtype=float)
        waypoint = np.array([target[0], target[1], max(target[2] + cfg.pre_grasp_z_offset, cfg.pre_grasp_z_min)])
        dist = float(np.linalg.norm(np.asarray(obs["ee_pos"], dtype=float) - waypoint))
        action = np.array([waypoint[0], waypoint[1], waypoint[2], 0.0])
        obs, _, done, info = env.step(action)
        if dist < cfg.approach_dist_threshold:
            return obs, NodeResult(completed=True, timed_out=False, steps_used=step + 1)
        if done:
            return obs, NodeResult(completed=False, timed_out=False, steps_used=step + 1, aborted=True)
    return obs, NodeResult(completed=False, timed_out=True, steps_used=cfg.timeout_approach)


def _run_lower(
    env: PandaPickPlaceEnv,
    obs: dict[str, Any],
    cfg: NodeConfig,
    logger: StructuredLogger,
) -> tuple[dict[str, Any], NodeResult]:
    """Descend to grasp height."""
    for step in range(cfg.timeout_lower):
        target = np.asarray(obs["target_pos"], dtype=float)
        grasp_pos = np.array([target[0], target[1], max(target[2] + cfg.grasp_z_offset, cfg.grasp_z_min)])
        dist = float(np.linalg.norm(np.asarray(obs["ee_pos"], dtype=float) - grasp_pos))
        contact = bool(obs["contacts_summary"]["target_hand_contact"])
        action = np.array([grasp_pos[0], grasp_pos[1], grasp_pos[2], 0.0])
        obs, _, done, info = env.step(action)
        if dist < cfg.descend_dist_threshold or contact:
            return obs, NodeResult(completed=True, timed_out=False, steps_used=step + 1)
        if done:
            return obs, NodeResult(completed=False, timed_out=False, steps_used=step + 1, aborted=True)
    return obs, NodeResult(completed=False, timed_out=True, steps_used=cfg.timeout_lower)


def _run_close(
    env: PandaPickPlaceEnv,
    obs: dict[str, Any],
    cfg: NodeConfig,
    logger: StructuredLogger,
) -> tuple[dict[str, Any], NodeResult]:
    """Close gripper and wait for weld attachment."""
    target = np.asarray(obs["target_pos"], dtype=float)
    grasp_pos = np.array([target[0], target[1], max(target[2] + cfg.grasp_z_offset, cfg.grasp_z_min)])
    for step in range(cfg.timeout_close):
        action = np.array([grasp_pos[0], grasp_pos[1], grasp_pos[2], 1.0])
        obs, _, done, info = env.step(action)
        if obs["gripper_state"]["attached"]:
            return obs, NodeResult(completed=True, timed_out=False, steps_used=step + 1)
        if done:
            return obs, NodeResult(completed=False, timed_out=False, steps_used=step + 1, aborted=True)
    # Timed out without attachment — request retry from approach
    return obs, NodeResult(completed=False, timed_out=True, steps_used=cfg.timeout_close, retry_approach=True)


def _run_lift(
    env: PandaPickPlaceEnv,
    obs: dict[str, Any],
    cfg: NodeConfig,
    logger: StructuredLogger,
) -> tuple[dict[str, Any], NodeResult]:
    """Lift the grasped cube to lift height."""
    ee = np.asarray(obs["ee_pos"], dtype=float)
    lift_target = np.array([ee[0], ee[1], cfg.lift_height_z])
    floor_clear_steps = 0
    for step in range(cfg.timeout_lift):
        target = np.asarray(obs["target_pos"], dtype=float)
        ee = np.asarray(obs["ee_pos"], dtype=float)
        attached = bool(obs["gripper_state"]["attached"])
        target_on_floor = bool(target[2] < 0.045)
        target_floor_contact = bool(obs["contacts_summary"]["target_floor_contact"])

        if not attached and target_on_floor:
            # Cube dropped — abort this node, request retry
            return obs, NodeResult(completed=False, timed_out=False, steps_used=step + 1, retry_approach=True)

        floor_clear_steps = floor_clear_steps + 1 if not target_floor_contact else 0
        action = np.array([lift_target[0], lift_target[1], lift_target[2], 1.0])
        obs, _, done, info = env.step(action)

        if (
            ee[2] >= cfg.ee_lift_clearance_z
            and target[2] >= cfg.target_lift_clearance_z
            and floor_clear_steps >= cfg.floor_clear_required
        ):
            return obs, NodeResult(completed=True, timed_out=False, steps_used=step + 1)
        if done:
            return obs, NodeResult(completed=False, timed_out=False, steps_used=step + 1, aborted=True)
    return obs, NodeResult(completed=False, timed_out=True, steps_used=cfg.timeout_lift)


def _run_move_to_goal(
    env: PandaPickPlaceEnv,
    obs: dict[str, Any],
    cfg: NodeConfig,
    logger: StructuredLogger,
) -> tuple[dict[str, Any], NodeResult]:
    """Transport cube to goal region at lift height."""
    for step in range(cfg.timeout_move):
        target = np.asarray(obs["target_pos"], dtype=float)
        goal = np.asarray(obs["goal_pos"], dtype=float)
        ee = np.asarray(obs["ee_pos"], dtype=float)
        attached = bool(obs["gripper_state"]["attached"])
        target_on_floor = bool(target[2] < 0.045)
        target_floor_contact = bool(obs["contacts_summary"]["target_floor_contact"])

        if not attached and target_on_floor:
            return obs, NodeResult(completed=False, timed_out=False, steps_used=step + 1, aborted=True)

        # If cube touches floor during transport, try to re-lift
        if target_floor_contact:
            lift_cmd = np.array([ee[0], ee[1], cfg.lift_height_z, 1.0])
            obs, _, done, _ = env.step(lift_cmd)
            if done:
                return obs, NodeResult(completed=False, timed_out=False, steps_used=step + 1, aborted=True)
            continue

        pre_place = np.array([goal[0], goal[1], cfg.lift_height_z])
        action = np.array([pre_place[0], pre_place[1], pre_place[2], 1.0])
        obs, _, done, info = env.step(action)

        target_goal_xy = float(np.linalg.norm(target[:2] - goal[:2]))
        if target_goal_xy < cfg.target_goal_xy_threshold:
            return obs, NodeResult(completed=True, timed_out=False, steps_used=step + 1)
        if done:
            return obs, NodeResult(completed=False, timed_out=False, steps_used=step + 1, aborted=True)
    return obs, NodeResult(completed=False, timed_out=True, steps_used=cfg.timeout_move)


def _run_open_gripper(
    env: PandaPickPlaceEnv,
    obs: dict[str, Any],
    cfg: NodeConfig,
    logger: StructuredLogger,
) -> tuple[dict[str, Any], NodeResult]:
    """Lower to place height, then open gripper and hold."""
    goal = np.asarray(obs["goal_pos"], dtype=float)
    place_pos = np.array([goal[0], goal[1], cfg.place_z])
    open_hold = 0
    lowered = False
    for step in range(cfg.timeout_open):
        ee = np.asarray(obs["ee_pos"], dtype=float)
        if not lowered:
            # First phase: lower to place height with gripper closed
            action = np.array([place_pos[0], place_pos[1], place_pos[2], 1.0])
            obs, _, done, info = env.step(action)
            if ee[2] < cfg.lower_to_open_z:
                lowered = True
        else:
            # Second phase: open gripper and hold
            action = np.array([place_pos[0], place_pos[1], place_pos[2], 0.0])
            obs, _, done, info = env.step(action)
            open_hold += 1
            if open_hold >= cfg.open_hold_required:
                return obs, NodeResult(completed=True, timed_out=False, steps_used=step + 1)
        if done:
            return obs, NodeResult(completed=False, timed_out=False, steps_used=step + 1, aborted=True)
    return obs, NodeResult(completed=False, timed_out=True, steps_used=cfg.timeout_open)


def _run_stabilize(
    env: PandaPickPlaceEnv,
    obs: dict[str, Any],
    cfg: NodeConfig,
    logger: StructuredLogger,
) -> tuple[dict[str, Any], NodeResult]:
    """Hold position for N steps."""
    ee = np.asarray(obs["ee_pos"], dtype=float)
    gripper_flag = 1.0 if obs["gripper_state"]["closed"] else 0.0
    for step in range(cfg.stabilize_steps):
        action = np.array([ee[0], ee[1], ee[2], gripper_flag])
        obs, _, done, _ = env.step(action)
        if done:
            return obs, NodeResult(completed=True, timed_out=False, steps_used=step + 1)
    return obs, NodeResult(completed=True, timed_out=False, steps_used=cfg.stabilize_steps)


# ---------------------------------------------------------------------------
# Node dispatch
# ---------------------------------------------------------------------------

_NODE_HANDLERS = {
    "approach_target": _run_approach,
    "lower_to_grasp": _run_lower,
    "close_gripper": _run_close,
    "lift_target": _run_lift,
    "move_to_goal": _run_move_to_goal,
    "open_gripper": _run_open_gripper,
    "stabilize": _run_stabilize,
}


# ---------------------------------------------------------------------------
# Graph execution
# ---------------------------------------------------------------------------

def _build_execution_order(graph: dict[str, Any]) -> list[dict[str, Any]]:
    """Walk edges to produce a linear execution order of nodes."""
    nodes_by_id = {n["id"]: n for n in graph["nodes"]}
    edges = graph.get("edges", [])
    if not edges:
        return list(graph["nodes"])

    # Find start node (not a target of any edge)
    targets = {e["to"] for e in edges}
    start_candidates = [n["id"] for n in graph["nodes"] if n["id"] not in targets]
    current_id = start_candidates[0] if start_candidates else graph["nodes"][0]["id"]

    # Build adjacency
    adj: dict[str, str] = {}
    for e in edges:
        adj[e["from"]] = e["to"]

    order: list[dict[str, Any]] = []
    visited: set[str] = set()
    while current_id and current_id not in visited:
        visited.add(current_id)
        if current_id in nodes_by_id:
            order.append(nodes_by_id[current_id])
        current_id = adj.get(current_id)  # type: ignore[assignment]

    return order


def run_episode_from_graph(
    env: PandaPickPlaceEnv,
    graph: dict[str, Any],
    logger: StructuredLogger,
    *,
    max_steps: int = 500,
    node_config: NodeConfig | None = None,
) -> dict[str, Any]:
    """Execute one episode driven by a task graph.

    Returns a result dict with: success, steps_used, grasp_achieved, final_target_goal_dist.
    """
    cfg = node_config or NodeConfig()
    obs = env.reset(seed=env.config.seed)
    execution_order = _build_execution_order(graph)
    total_steps = 0
    grasp_achieved = False
    grasp_retries = 0

    logger.log_event("execution_start", {
        "seed": env.config.seed,
        "num_nodes": len(execution_order),
        "node_types": [n["type"] for n in execution_order],
    })

    node_idx = 0
    while node_idx < len(execution_order):
        node = execution_order[node_idx]
        node_type = node["type"]
        handler = _NODE_HANDLERS.get(node_type)

        if handler is None:
            logger.log_event("unknown_node_type", {"node": node}, step=total_steps)
            node_idx += 1
            continue

        logger.log_event("node_start", {
            "node_id": node["id"],
            "node_type": node_type,
            "node_idx": node_idx,
        }, step=total_steps)

        obs, result = handler(env, obs, cfg, logger)
        total_steps += result.steps_used

        if obs["gripper_state"]["attached"]:
            grasp_achieved = True

        logger.log_event("node_end", {
            "node_id": node["id"],
            "node_type": node_type,
            "completed": result.completed,
            "timed_out": result.timed_out,
            "steps_used": result.steps_used,
            "aborted": result.aborted,
            "retry_approach": result.retry_approach,
        }, step=total_steps)

        if result.aborted:
            break

        if result.retry_approach and node_type == "close_gripper":
            grasp_retries += 1
            if grasp_retries > cfg.max_grasp_retries:
                logger.log_event("grasp_retries_exhausted", {"retries": grasp_retries}, step=total_steps)
                break
            # Rewind to first approach_target node
            approach_idx = None
            for i, n in enumerate(execution_order):
                if n["type"] == "approach_target":
                    approach_idx = i
                    break
            if approach_idx is not None:
                logger.log_event("grasp_retry", {"retry": grasp_retries, "rewind_to": approach_idx}, step=total_steps)
                node_idx = approach_idx
                continue
            else:
                break

        if total_steps >= max_steps:
            logger.log_event("max_steps_reached", {"total_steps": total_steps}, step=total_steps)
            break

        node_idx += 1

    # Final observations
    target = np.asarray(obs["target_pos"], dtype=float)
    goal = np.asarray(obs["goal_pos"], dtype=float)
    final_dist = float(np.linalg.norm(target[:2] - goal[:2]))
    success = bool(
        final_dist < 0.06
        and target[2] < 0.07
        and not obs["gripper_state"]["attached"]
        and obs["gripper_state"]["width"] > 0.008
    )

    result_dict = {
        "success": success,
        "steps_used": total_steps,
        "grasp_achieved": grasp_achieved,
        "grasp_retries": grasp_retries,
        "final_target_goal_dist": final_dist,
        "final_target_z": float(target[2]),
        "nodes_executed": node_idx + 1,
        "nodes_total": len(execution_order),
    }
    logger.log_event("execution_done", result_dict, step=total_steps)
    return result_dict


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute a task graph on PandaPickPlaceEnv.")
    parser.add_argument("--task-graph", type=Path, required=True, help="Path to task_graph.json.")
    parser.add_argument("--seed", type=int, default=0, help="Environment seed.")
    parser.add_argument("--steps", type=int, default=500, help="Max steps per episode.")
    render_group = parser.add_mutually_exclusive_group()
    render_group.add_argument("--render", action="store_true", help="Enable MuJoCo viewer.")
    render_group.add_argument("--headless", action="store_true", help="Force headless mode.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/execution"), help="Artifact directory.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    with args.task_graph.open("r", encoding="utf-8") as f:
        graph = json.load(f)

    render_mode = bool(args.render and not args.headless)
    config = EnvConfig(
        seed=args.seed,
        render=render_mode,
        max_steps=args.steps,
        output_dir=args.output_dir,
        sanity_asserts=False,
    )
    logger = StructuredLogger(output_dir=args.output_dir, run_name="execution", enabled=True)

    with PandaPickPlaceEnv(config) as env:
        result = run_episode_from_graph(env, graph, logger, max_steps=args.steps)

    logger.save_json("execution_result.json", result)
    logger.close()

    status = "SUCCESS" if result["success"] else "FAIL"
    print(
        f"[{status}] seed={args.seed} | steps={result['steps_used']} | "
        f"grasp={result['grasp_achieved']} | tgt-goal={result['final_target_goal_dist']:.3f} | "
        f"nodes={result['nodes_executed']}/{result['nodes_total']}"
    )


if __name__ == "__main__":
    main()
