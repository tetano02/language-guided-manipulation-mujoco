"""Execution engine: interpret a task graph and drive PandaPickPlaceEnv with dynamic sub-goals."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

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


# Frame capture callback type: called after each env.step with node info
FrameCallback = Callable[[PandaPickPlaceEnv, str, int], None]

_NOOP_FRAME: FrameCallback = lambda env, node_type, step: None


# ---------------------------------------------------------------------------
# Human-readable node labels + emoji
# ---------------------------------------------------------------------------

_NODE_LABELS: dict[str, str] = {
    "approach_target": "🔵 APPROACH TARGET",
    "lower_to_grasp":  "🔵 LOWER TO GRASP",
    "close_gripper":   "🟢 CLOSE GRIPPER",
    "lift_target":     "🟡 LIFT TARGET",
    "move_to_goal":    "🟡 MOVE TO GOAL",
    "open_gripper":    "🔴 OPEN GRIPPER",
    "stabilize":       "⚪ STABILIZE",
}


# ---------------------------------------------------------------------------
# Individual node handlers
# ---------------------------------------------------------------------------

def _run_approach(
    env: PandaPickPlaceEnv,
    obs: dict[str, Any],
    cfg: NodeConfig,
    logger: StructuredLogger,
    on_frame: FrameCallback = _NOOP_FRAME,
) -> tuple[dict[str, Any], NodeResult]:
    """Move EE above the target cube."""
    for step in range(cfg.timeout_approach):
        target = np.asarray(obs["target_pos"], dtype=float)
        waypoint = np.array([target[0], target[1], max(target[2] + cfg.pre_grasp_z_offset, cfg.pre_grasp_z_min)])
        dist = float(np.linalg.norm(np.asarray(obs["ee_pos"], dtype=float) - waypoint))
        action = np.array([waypoint[0], waypoint[1], waypoint[2], 0.0])
        obs, _, done, info = env.step(action)
        on_frame(env, "approach_target", step)
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
    on_frame: FrameCallback = _NOOP_FRAME,
) -> tuple[dict[str, Any], NodeResult]:
    """Descend to grasp height."""
    for step in range(cfg.timeout_lower):
        target = np.asarray(obs["target_pos"], dtype=float)
        grasp_pos = np.array([target[0], target[1], max(target[2] + cfg.grasp_z_offset, cfg.grasp_z_min)])
        dist = float(np.linalg.norm(np.asarray(obs["ee_pos"], dtype=float) - grasp_pos))
        contact = bool(obs["contacts_summary"]["target_hand_contact"])
        action = np.array([grasp_pos[0], grasp_pos[1], grasp_pos[2], 0.0])
        obs, _, done, info = env.step(action)
        on_frame(env, "lower_to_grasp", step)
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
    on_frame: FrameCallback = _NOOP_FRAME,
) -> tuple[dict[str, Any], NodeResult]:
    """Close gripper and wait for weld attachment."""
    target = np.asarray(obs["target_pos"], dtype=float)
    grasp_pos = np.array([target[0], target[1], max(target[2] + cfg.grasp_z_offset, cfg.grasp_z_min)])
    for step in range(cfg.timeout_close):
        action = np.array([grasp_pos[0], grasp_pos[1], grasp_pos[2], 1.0])
        obs, _, done, info = env.step(action)
        on_frame(env, "close_gripper", step)
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
    on_frame: FrameCallback = _NOOP_FRAME,
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
            return obs, NodeResult(completed=False, timed_out=False, steps_used=step + 1, retry_approach=True)

        floor_clear_steps = floor_clear_steps + 1 if not target_floor_contact else 0
        action = np.array([lift_target[0], lift_target[1], lift_target[2], 1.0])
        obs, _, done, info = env.step(action)
        on_frame(env, "lift_target", step)

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
    on_frame: FrameCallback = _NOOP_FRAME,
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

        if target_floor_contact:
            lift_cmd = np.array([ee[0], ee[1], cfg.lift_height_z, 1.0])
            obs, _, done, _ = env.step(lift_cmd)
            on_frame(env, "move_to_goal", step)
            if done:
                return obs, NodeResult(completed=False, timed_out=False, steps_used=step + 1, aborted=True)
            continue

        pre_place = np.array([goal[0], goal[1], cfg.lift_height_z])
        action = np.array([pre_place[0], pre_place[1], pre_place[2], 1.0])
        obs, _, done, info = env.step(action)
        on_frame(env, "move_to_goal", step)

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
    on_frame: FrameCallback = _NOOP_FRAME,
) -> tuple[dict[str, Any], NodeResult]:
    """Lower to place height, then open gripper and hold."""
    goal = np.asarray(obs["goal_pos"], dtype=float)
    place_pos = np.array([goal[0], goal[1], cfg.place_z])
    open_hold = 0
    lowered = False
    for step in range(cfg.timeout_open):
        ee = np.asarray(obs["ee_pos"], dtype=float)
        if not lowered:
            action = np.array([place_pos[0], place_pos[1], place_pos[2], 1.0])
            obs, _, done, info = env.step(action)
            on_frame(env, "open_gripper", step)
            if ee[2] < cfg.lower_to_open_z:
                lowered = True
        else:
            action = np.array([place_pos[0], place_pos[1], place_pos[2], 0.0])
            obs, _, done, info = env.step(action)
            on_frame(env, "open_gripper", step)
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
    on_frame: FrameCallback = _NOOP_FRAME,
) -> tuple[dict[str, Any], NodeResult]:
    """Hold position for N steps."""
    ee = np.asarray(obs["ee_pos"], dtype=float)
    gripper_flag = 1.0 if obs["gripper_state"]["closed"] else 0.0
    for step in range(cfg.stabilize_steps):
        action = np.array([ee[0], ee[1], ee[2], gripper_flag])
        obs, _, done, _ = env.step(action)
        on_frame(env, "stabilize", step)
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

    targets = {e["to"] for e in edges}
    start_candidates = [n["id"] for n in graph["nodes"] if n["id"] not in targets]
    current_id = start_candidates[0] if start_candidates else graph["nodes"][0]["id"]

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
    on_frame: FrameCallback | None = None,
) -> dict[str, Any]:
    """Execute one episode driven by a task graph.

    Returns a result dict with: success, steps_used, grasp_achieved, final_target_goal_dist.
    """
    cfg = node_config or NodeConfig()
    frame_cb = on_frame or _NOOP_FRAME
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

        obs, result = handler(env, obs, cfg, logger, on_frame=frame_cb)
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
# Video rendering with node overlay
# ---------------------------------------------------------------------------

def _add_text_overlay(frame: np.ndarray, text: str, step: int) -> np.ndarray:
    """Add text overlay to a video frame using PIL."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    # Try to use a nice font, fall back to default
    font_size = max(20, frame.shape[0] // 25)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

    small_font_size = max(14, frame.shape[0] // 35)
    try:
        small_font = ImageFont.truetype("arial.ttf", small_font_size)
    except (OSError, IOError):
        try:
            small_font = ImageFont.truetype("DejaVuSans.ttf", small_font_size)
        except (OSError, IOError):
            small_font = ImageFont.load_default()

    # Draw semi-transparent background bar at top
    bar_height = font_size + small_font_size + 24
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle([(0, 0), (img.width, bar_height)], fill=(0, 0, 0, 160))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Node name (large, top)
    draw.text((12, 6), text, fill=(255, 255, 255), font=font)

    # Step counter (smaller, below)
    draw.text((12, font_size + 10), f"Step {step}", fill=(200, 200, 200), font=small_font)

    return np.array(img)


def _save_video(frames: list[np.ndarray], output_path: Path, fps: int) -> bool:
    """Save frames as an MP4 video."""
    if not frames:
        return False
    try:
        import imageio.v2 as imageio
    except Exception:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(output_path), fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute a task graph on PandaPickPlaceEnv.")
    parser.add_argument("--task-graph", type=Path, required=True, help="Path to task_graph.json.")
    parser.add_argument("--seed", type=int, default=0, help="Environment seed (or base seed with --multi-seed).")
    parser.add_argument("--steps", type=int, default=500, help="Max steps per episode.")
    render_group = parser.add_mutually_exclusive_group()
    render_group.add_argument("--render", action="store_true", help="Enable MuJoCo viewer.")
    render_group.add_argument("--headless", action="store_true", help="Force headless mode.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/execution"), help="Artifact directory.")
    # Video options
    parser.add_argument("--save-mp4", action="store_true", help="Save video with node overlay.")
    parser.add_argument("--video-fps", type=int, default=30, help="Video FPS.")
    parser.add_argument("--video-width", type=int, default=1280, help="Video width.")
    parser.add_argument("--video-height", type=int, default=960, help="Video height.")
    parser.add_argument("--video-every", type=int, default=1, help="Capture every N-th step.")
    # Multi-seed for generalization demo
    parser.add_argument("--multi-seed", type=int, default=0,
                        help="If > 0, run this many seeds starting from --seed and save one video each.")
    return parser


def main() -> None:
    import uuid
    from datetime import datetime

    args = _build_parser().parse_args()
    with args.task_graph.open("r", encoding="utf-8") as f:
        graph = json.load(f)

    seeds = [args.seed] if args.multi_seed <= 0 else [args.seed + i for i in range(args.multi_seed)]
    is_multi = args.multi_seed > 0

    # ── Build output directory: artifacts/execution/YYYY-MM-DD/<run_folder>/
    now = datetime.now()
    date_dir = args.output_dir / now.strftime("%Y-%m-%d")
    date_dir.mkdir(parents=True, exist_ok=True)

    if is_multi:
        uid = uuid.uuid4().hex[:6]
        run_dir = date_dir / f"demo_{now.strftime('%H%M%S')}_{uid}"
    else:
        run_dir = date_dir / f"run_{now.strftime('%H%M%S')}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        render_mode = bool(args.render and not args.headless)
        # Disable env-level logging to avoid extra env_* folders
        config = EnvConfig(
            seed=seed,
            render=render_mode,
            max_steps=args.steps,
            output_dir=run_dir,
            sanity_asserts=False,
            log_jsonl=False,
        )
        # Use a silent logger (we save results manually to run_dir)
        logger = StructuredLogger(output_dir=run_dir, run_name="log", enabled=False)

        # Set up video capture if requested
        import mujoco as mj
        frames: list[np.ndarray] = []
        step_counter = [0]

        if args.save_mp4:
            try:
                with PandaPickPlaceEnv(config) as env:
                    renderer = mj.Renderer(env.model, height=args.video_height, width=args.video_width)

                    def on_frame(env: PandaPickPlaceEnv, node_type: str, node_step: int) -> None:
                        step_counter[0] += 1
                        if step_counter[0] % max(args.video_every, 1) != 0:
                            return
                        renderer.update_scene(env.data)
                        raw = renderer.render().copy()
                        label = _NODE_LABELS.get(node_type, node_type)
                        annotated = _add_text_overlay(raw, label, step_counter[0])
                        frames.append(annotated)

                    result = run_episode_from_graph(env, graph, logger, max_steps=args.steps, on_frame=on_frame)

                    # Add a few extra frames at the end to show final state
                    final_label = "✅ SUCCESS" if result["success"] else "❌ FAIL"
                    for _ in range(args.video_fps):
                        renderer.update_scene(env.data)
                        raw = renderer.render().copy()
                        annotated = _add_text_overlay(raw, final_label, step_counter[0])
                        frames.append(annotated)

                    renderer.close()

                if frames:
                    status_tag = "ok" if result["success"] else "fail"
                    video_name = f"seed{seed}_{status_tag}_{result['steps_used']}steps.mp4"
                    video_path = run_dir / video_name
                    saved = _save_video(frames, video_path, args.video_fps)
                    if saved:
                        print(f"  Video: {video_path}")
            except Exception as exc:
                print(f"  Video error: {exc}")
                with PandaPickPlaceEnv(config) as env:
                    result = run_episode_from_graph(env, graph, logger, max_steps=args.steps)
        else:
            with PandaPickPlaceEnv(config) as env:
                result = run_episode_from_graph(env, graph, logger, max_steps=args.steps)

        # Save result JSON
        result_name = f"result_seed{seed}.json" if is_multi else "result.json"
        (run_dir / result_name).write_text(json.dumps(result, indent=2))

        status = "SUCCESS" if result["success"] else "FAIL"
        print(
            f"[{status}] seed={seed} | steps={result['steps_used']} | "
            f"grasp={result['grasp_achieved']} | tgt-goal={result['final_target_goal_dist']:.3f} | "
            f"nodes={result['nodes_executed']}/{result['nodes_total']}"
        )

    print(f"\nOutput: {run_dir}")


if __name__ == "__main__":
    main()

