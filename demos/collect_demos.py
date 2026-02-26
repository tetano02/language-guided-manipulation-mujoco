"""Scripted deterministic demo collection for Panda pick-and-place."""

from __future__ import annotations

import argparse
import copy
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from mujoco_env.env import EnvConfig, PandaPickPlaceEnv
from utils.logging import StructuredLogger


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect scripted pick-and-place demos.")
    render_group = parser.add_mutually_exclusive_group()
    render_group.add_argument("--render", action="store_true", help="Enable MuJoCo viewer.")
    render_group.add_argument("--headless", action="store_true", help="Force headless mode.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to collect.")
    parser.add_argument("--steps", type=int, default=400, help="Max steps per episode.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed; episode seed = base + episode index.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/demos"), help="Artifact directory.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("demos/dataset.pkl"),
        help="Output path for dataset pickle.",
    )
    parser.add_argument(
        "--no-sanity-asserts",
        action="store_true",
        help="Do not raise on sanity violations (still logged).",
    )
    parser.add_argument("--save-mp4", action="store_true", help="Optionally save one mp4 per episode.")
    parser.add_argument("--video-fps", type=int, default=30, help="FPS for optional mp4.")
    parser.add_argument("--video-every", type=int, default=2, help="Capture one frame every N env steps.")
    parser.add_argument("--video-width", type=int, default=640, help="Optional mp4 width.")
    parser.add_argument("--video-height", type=int, default=480, help="Optional mp4 height.")
    return parser


def _transition(
    logger: StructuredLogger,
    *,
    episode_idx: int,
    step_idx: int,
    current_phase: str,
    next_phase: str,
) -> str:
    if current_phase == next_phase:
        return current_phase
    logger.log_event(
        "phase_transition",
        {
            "episode_idx": episode_idx,
            "from_phase": current_phase,
            "to_phase": next_phase,
        },
        step=step_idx,
    )
    return next_phase


def _save_episode_video(
    frames: list[np.ndarray],
    *,
    output_path: Path,
    fps: int,
) -> bool:
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


def main() -> None:
    args = _build_parser().parse_args()
    render_mode = bool(args.render and not args.headless)
    config = EnvConfig(
        seed=args.seed,
        render=render_mode,
        output_dir=args.output_dir,
        sanity_asserts=not args.no_sanity_asserts,
    )

    logger = StructuredLogger(output_dir=args.output_dir, run_name="collect_demos", enabled=True)
    dataset: list[list[dict[str, Any]]] = []
    summaries: list[dict[str, Any]] = []

    with PandaPickPlaceEnv(config) as env:
        renderer: Any | None = None
        if args.save_mp4:
            try:
                import mujoco

                renderer = mujoco.Renderer(env.model, height=args.video_height, width=args.video_width)
            except Exception as exc:
                logger.log_event("video_unavailable", {"error": repr(exc)})
                renderer = None

        for episode_idx in range(args.episodes):
            episode_seed = int(args.seed + episode_idx)
            obs = env.reset(seed=episode_seed)
            start_ee = np.array(obs["ee_pos"], dtype=float)
            phase = "approach"
            retry_count = 0
            close_hold_steps = 0
            open_hold_steps = 0
            max_pick_retries = 3
            floor_clear_steps = 0
            floor_clear_required = 6
            ee_lift_clearance_z = 0.06
            target_lift_clearance_z = 0.04
            lift_height_z = 0.28
            lift_anchor_xy = np.array(start_ee[:2], dtype=float)
            attached_once = False
            episode_states: list[dict[str, Any]] = []
            frames: list[np.ndarray] = []
            final_info: dict[str, Any] = {}
            final_reward = 0.0

            logger.log_event(
                "episode_start",
                {"episode_idx": episode_idx, "seed": episode_seed},
            )

            for t in range(args.steps):
                target = np.array(obs["target_pos"], dtype=float)
                goal = np.array(obs["goal_pos"], dtype=float)
                ee = np.array(obs["ee_pos"], dtype=float)
                attached = bool(obs["gripper_state"]["attached"])
                target_on_floor = bool(target[2] < 0.045)
                target_floor_contact = bool(obs["contacts_summary"]["target_floor_contact"])

                pre_grasp = np.array([target[0], target[1], max(target[2] + 0.16, 0.18)], dtype=float)
                grasp = np.array([target[0], target[1], max(target[2] + 0.035, 0.055)], dtype=float)
                lift = np.array([lift_anchor_xy[0], lift_anchor_xy[1], lift_height_z], dtype=float)
                pre_place = np.array([goal[0], goal[1], lift_height_z], dtype=float)
                place = np.array([goal[0], goal[1], 0.08], dtype=float)

                dist_to_pre_grasp = float(np.linalg.norm(ee - pre_grasp))
                dist_to_grasp = float(np.linalg.norm(ee - grasp))
                target_goal_xy = float(np.linalg.norm(target[:2] - goal[:2]))

                if phase == "approach":
                    ee_cmd = pre_grasp
                    gripper = 0.0
                    if dist_to_pre_grasp < 0.045:
                        phase = _transition(
                            logger,
                            episode_idx=episode_idx,
                            step_idx=t + 1,
                            current_phase=phase,
                            next_phase="descend",
                        )
                elif phase == "descend":
                    ee_cmd = grasp
                    gripper = 0.0
                    if dist_to_grasp < 0.035 or obs["contacts_summary"]["target_hand_contact"]:
                        phase = _transition(
                            logger,
                            episode_idx=episode_idx,
                            step_idx=t + 1,
                            current_phase=phase,
                            next_phase="close",
                        )
                elif phase == "close":
                    ee_cmd = grasp
                    gripper = 1.0
                    close_hold_steps += 1
                    if attached:
                        lift_anchor_xy = np.array(start_ee[:2], dtype=float)
                        floor_clear_steps = 0
                        phase = _transition(
                            logger,
                            episode_idx=episode_idx,
                            step_idx=t + 1,
                            current_phase=phase,
                            next_phase="lift_vertical",
                        )
                    elif close_hold_steps > 20:
                        retry_count += 1
                        next_phase = "abort" if retry_count > max_pick_retries else "approach"
                        phase = _transition(
                            logger,
                            episode_idx=episode_idx,
                            step_idx=t + 1,
                            current_phase=phase,
                            next_phase=next_phase,
                        )
                        close_hold_steps = 0
                        floor_clear_steps = 0
                elif phase == "lift_vertical":
                    ee_cmd = lift
                    gripper = 1.0
                    if not attached and target_on_floor:
                        phase = _transition(
                            logger,
                            episode_idx=episode_idx,
                            step_idx=t + 1,
                            current_phase=phase,
                            next_phase="approach",
                        )
                    else:
                        floor_clear_steps = floor_clear_steps + 1 if not target_floor_contact else 0
                        if (
                            ee[2] >= ee_lift_clearance_z
                            and target[2] >= target_lift_clearance_z
                            and floor_clear_steps >= floor_clear_required
                        ):
                            phase = _transition(
                                logger,
                                episode_idx=episode_idx,
                                step_idx=t + 1,
                                current_phase=phase,
                                next_phase="move_to_goal_high",
                            )
                elif phase == "move_to_goal_high":
                    ee_cmd = pre_place
                    gripper = 1.0
                    if not attached and target_on_floor:
                        phase = _transition(
                            logger,
                            episode_idx=episode_idx,
                            step_idx=t + 1,
                            current_phase=phase,
                            next_phase="approach",
                        )
                    elif target_floor_contact:
                        phase = _transition(
                            logger,
                            episode_idx=episode_idx,
                            step_idx=t + 1,
                            current_phase=phase,
                            next_phase="lift_vertical",
                        )
                    elif target_goal_xy < 0.08:
                        phase = _transition(
                            logger,
                            episode_idx=episode_idx,
                            step_idx=t + 1,
                            current_phase=phase,
                            next_phase="lower_to_place",
                        )
                elif phase == "lower_to_place":
                    ee_cmd = place
                    gripper = 1.0
                    if ee[2] < 0.12:
                        phase = _transition(
                            logger,
                            episode_idx=episode_idx,
                            step_idx=t + 1,
                            current_phase=phase,
                            next_phase="open",
                        )
                elif phase == "open":
                    ee_cmd = place
                    gripper = 0.0
                    open_hold_steps += 1
                    if open_hold_steps > 18:
                        phase = _transition(
                            logger,
                            episode_idx=episode_idx,
                            step_idx=t + 1,
                            current_phase=phase,
                            next_phase="retreat",
                        )
                elif phase == "retreat":
                    ee_cmd = pre_place
                    gripper = 0.0
                else:
                    ee_cmd = np.array([start_ee[0], start_ee[1], 0.24], dtype=float)
                    gripper = 0.0

                action = np.array([ee_cmd[0], ee_cmd[1], ee_cmd[2], gripper], dtype=float)
                obs, reward, done, info = env.step(action)
                final_info = dict(info)
                final_reward = float(reward)

                if info["attachment_changed"] and info["grasp_attached"]:
                    attached_once = True

                state = copy.deepcopy(obs)
                state["info"] = copy.deepcopy(info)
                state["phase"] = phase
                state["episode_idx"] = episode_idx
                state["timestep"] = t + 1
                episode_states.append(state)

                if renderer is not None and ((t + 1) % max(args.video_every, 1) == 0):
                    renderer.update_scene(env.data)
                    frames.append(renderer.render().copy())

                if phase == "abort":
                    logger.log_event(
                        "episode_abort",
                        {
                            "episode_idx": episode_idx,
                            "reason": "max_pick_retries_exceeded",
                            "retry_count": retry_count,
                        },
                        step=t + 1,
                    )
                    done = True

                if done:
                    break

            dataset.append(episode_states)
            summary = {
                "episode_idx": episode_idx,
                "seed": episode_seed,
                "num_steps": len(episode_states),
                "final_phase": phase,
                "final_reward": final_reward,
                "final_success": bool(final_info.get("success", False)),
                "grasp_attached_once": attached_once,
                "final_grasp_attached": bool(final_info.get("grasp_attached", False)),
                "retry_count": retry_count,
                "sanity_ok_last": bool(final_info.get("sanity_ok", True)),
            }

            if args.save_mp4:
                video_path = logger.run_dir / f"episode_{episode_idx:03d}_seed{episode_seed}_steps{len(episode_states)}.mp4"
                summary["video_path"] = str(video_path)
                summary["video_saved"] = bool(_save_episode_video(frames, output_path=video_path, fps=args.video_fps))

            summaries.append(summary)
            logger.log_event("episode_done", summary)

        if renderer is not None:
            renderer.close()

    args.dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with args.dataset_path.open("wb") as handle:
        pickle.dump(dataset, handle)

    logger.save_pickle("dataset.pkl", dataset)
    logger.save_json("episodes_summary.json", summaries)
    logger.log_event(
        "dataset_saved",
        {
            "dataset_path": str(args.dataset_path),
            "episodes": len(dataset),
            "output_dir": str(logger.run_dir),
        },
    )
    logger.close()


if __name__ == "__main__":
    main()
