"""Debug a single episode to understand lift_vertical failure."""
from __future__ import annotations
import sys
import numpy as np
sys.path.insert(0, ".")
from mujoco_env.env import EnvConfig, PandaPickPlaceEnv


def run_debug(seed: int = 0, max_steps: int = 200):
    config = EnvConfig(seed=seed, render=False, max_steps=max_steps, sanity_asserts=False, log_jsonl=False)
    env = PandaPickPlaceEnv(config)
    obs = env.reset(seed=seed)
    start_ee = np.array(obs["ee_pos"], dtype=float)
    phase = "approach"
    close_hold_steps = 0
    floor_clear_steps = 0
    lift_height_z = 0.28
    lift_anchor_xy = np.array(start_ee[:2], dtype=float)

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

        dist_to_pre = float(np.linalg.norm(ee - pre_grasp))
        dist_to_grasp = float(np.linalg.norm(ee - grasp))

        if phase == "approach":
            ee_cmd, gripper = pre_grasp, 0.0
            if dist_to_pre < 0.045: phase = "descend"
        elif phase == "descend":
            ee_cmd, gripper = grasp, 0.0
            if dist_to_grasp < 0.035 or obs["contacts_summary"]["target_hand_contact"]:
                phase = "close"
        elif phase == "close":
            ee_cmd, gripper = grasp, 1.0
            close_hold_steps += 1
            if attached:
                lift_anchor_xy = np.array(start_ee[:2], dtype=float)
                floor_clear_steps = 0
                phase = "lift_vertical"
            elif close_hold_steps > 20:
                phase = "abort"
        elif phase == "lift_vertical":
            ee_cmd, gripper = lift, 1.0
            if not attached and target_on_floor:
                phase = "approach"
            else:
                floor_clear_steps = floor_clear_steps + 1 if not target_floor_contact else 0
                if ee[2] >= 0.075 and target[2] >= 0.065 and floor_clear_steps >= 6:
                    phase = "move_to_goal_high"
        else:
            ee_cmd = np.array([start_ee[0], start_ee[1], 0.24], dtype=float)
            gripper = 0.0

        action = np.array([ee_cmd[0], ee_cmd[1], ee_cmd[2], gripper], dtype=float)
        obs, reward, done, info = env.step(action)

        if phase in ("lift_vertical", "close") or t < 5:
            weld_active = int(env.data.eq_active[env.controller.weld_id]) if hasattr(env.data, "eq_active") else -1
            pinch = env.controller._compute_pinch_center_world_pos()
            pinch_z = float(pinch[2]) if pinch is not None else -1.0
            print(
                f"t={t:3d} phase={phase:16s} ee_z={ee[2]:.4f} tgt_z={target[2]:.4f} "
                f"attached={attached} weld_active={weld_active} "
                f"floor_contact={target_floor_contact} pinch_z={pinch_z:.4f} "
                f"ee_cmd_z={ee_cmd[2]:.4f} floor_clear={floor_clear_steps}"
            )

        if done or phase == "abort":
            break

    env.close()
    print(f"\nFinal: phase={phase}, steps={t+1}")


if __name__ == "__main__":
    run_debug(seed=0)
