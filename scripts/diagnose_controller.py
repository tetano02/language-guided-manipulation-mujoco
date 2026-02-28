"""Diagnostic script: run scripted pick-and-place across seeds and report stats."""
from __future__ import annotations

import sys
import numpy as np

sys.path.insert(0, ".")
from mujoco_env.env import EnvConfig, PandaPickPlaceEnv


def run_episode(seed: int, max_steps: int = 400) -> dict:
    config = EnvConfig(
        seed=seed,
        render=False,
        max_steps=max_steps,
        sanity_asserts=False,
        log_jsonl=False,
    )
    env = PandaPickPlaceEnv(config)
    try:
        obs = env.reset(seed=seed)
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
        reached_goal_phase = False

        for t in range(max_steps):
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
                    phase = "descend"
            elif phase == "descend":
                ee_cmd = grasp
                gripper = 0.0
                if dist_to_grasp < 0.035 or obs["contacts_summary"]["target_hand_contact"]:
                    phase = "close"
            elif phase == "close":
                ee_cmd = grasp
                gripper = 1.0
                close_hold_steps += 1
                if attached:
                    lift_anchor_xy = np.array(start_ee[:2], dtype=float)
                    floor_clear_steps = 0
                    attached_once = True
                    phase = "lift_vertical"
                elif close_hold_steps > 20:
                    retry_count += 1
                    if retry_count > max_pick_retries:
                        phase = "abort"
                    else:
                        phase = "approach"
                    close_hold_steps = 0
                    floor_clear_steps = 0
            elif phase == "lift_vertical":
                ee_cmd = lift
                gripper = 1.0
                if not attached and target_on_floor:
                    phase = "approach"
                else:
                    floor_clear_steps = floor_clear_steps + 1 if not target_floor_contact else 0
                    if (
                        ee[2] >= ee_lift_clearance_z
                        and target[2] >= target_lift_clearance_z
                        and floor_clear_steps >= floor_clear_required
                    ):
                        phase = "move_to_goal_high"
            elif phase == "move_to_goal_high":
                ee_cmd = pre_place
                gripper = 1.0
                reached_goal_phase = True
                if not attached and target_on_floor:
                    phase = "approach"
                elif target_floor_contact:
                    phase = "lift_vertical"
                elif target_goal_xy < 0.08:
                    phase = "lower_to_place"
            elif phase == "lower_to_place":
                ee_cmd = place
                gripper = 1.0
                if ee[2] < 0.12:
                    phase = "open"
            elif phase == "open":
                ee_cmd = place
                gripper = 0.0
                open_hold_steps += 1
                if open_hold_steps > 18:
                    phase = "retreat"
            elif phase == "retreat":
                ee_cmd = pre_place
                gripper = 0.0
            else:
                ee_cmd = np.array([start_ee[0], start_ee[1], 0.24], dtype=float)
                gripper = 0.0

            action = np.array([ee_cmd[0], ee_cmd[1], ee_cmd[2], gripper], dtype=float)
            obs, reward, done, info = env.step(action)

            if phase == "abort":
                done = True
            if done:
                break

        target_final = np.array(obs["target_pos"], dtype=float)
        goal_final = np.array(obs["goal_pos"], dtype=float)
        ee_final = np.array(obs["ee_pos"], dtype=float)
        return {
            "seed": seed,
            "final_phase": phase,
            "success": bool(info.get("success", False)),
            "attached_once": attached_once,
            "retry_count": retry_count,
            "steps_used": t + 1,
            "target_goal_xy_dist": float(np.linalg.norm(target_final[:2] - goal_final[:2])),
            "target_z": float(target_final[2]),
            "ee_z": float(ee_final[2]),
            "reached_goal_phase": reached_goal_phase,
        }
    finally:
        env.close()


if __name__ == "__main__":
    seeds = list(range(20))
    results = []
    for s in seeds:
        r = run_episode(s)
        results.append(r)
        status = "OK" if r["success"] else "FAIL"
        print(
            f"Seed {s:3d}: {status:4s} | phase={r['final_phase']:20s} | "
            f"grasp={r['attached_once']} | retries={r['retry_count']} | "
            f"steps={r['steps_used']:3d} | tgt-goal={r['target_goal_xy_dist']:.3f} | "
            f"tgt_z={r['target_z']:.3f} | reached_goal={r['reached_goal_phase']}"
        )

    n_success = sum(1 for r in results if r["success"])
    n_grasp = sum(1 for r in results if r["attached_once"])
    n_goal = sum(1 for r in results if r["reached_goal_phase"])
    print(f"\n=== SUMMARY: {n_success}/{len(seeds)} success, {n_grasp}/{len(seeds)} grasped, {n_goal}/{len(seeds)} reached goal phase ===")
