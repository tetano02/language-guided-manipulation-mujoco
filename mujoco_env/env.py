"""Franka Panda pick-and-place environment with robust debug hooks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from mujoco_env.controller import EEController, EEControllerConfig
from utils.logging import StructuredLogger


def _name_or_fallback(model: mujoco.MjModel, obj_type: mujoco.mjtObj, obj_id: int) -> str:
    name = mujoco.mj_id2name(model, obj_type, obj_id)
    return name if name is not None else f"{obj_type.name}:{obj_id}"


@dataclass
class EnvConfig:
    model_path: Path = field(default_factory=lambda: Path(__file__).with_name("model.xml"))
    seed: int = 0
    render: bool = False
    frame_skip: int = 8
    max_steps: int = 350
    output_dir: Path = Path("artifacts")
    log_jsonl: bool = True
    sanity_asserts: bool = True
    ee_target_max_delta: float = 0.035
    workspace_low: np.ndarray = field(
        default_factory=lambda: np.array([0.42, -0.24, 0.05], dtype=float)
    )
    workspace_high: np.ndarray = field(
        default_factory=lambda: np.array([0.70, 0.24, 0.60], dtype=float)
    )


class PandaPickPlaceEnv:
    """EE-first MuJoCo environment for pick-and-place with two cubes."""

    def __init__(self, config: EnvConfig | None = None) -> None:
        self.config = config or EnvConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.model = mujoco.MjModel.from_xml_path(str(self.config.model_path))
        self.data = mujoco.MjData(self.model)
        self.controller = EEController(self.model, self.data, EEControllerConfig())
        self.logger = StructuredLogger(
            output_dir=self.config.output_dir,
            run_name="env",
            enabled=self.config.log_jsonl,
        )
        self.viewer: Any | None = None
        self._init_ids()
        self.step_count = 0
        self.last_action = np.zeros(4, dtype=float)
        self.current_ee_target = np.array([0.52, 0.0, 0.30], dtype=float)
        self.reference_ee_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        if self.config.render:
            self._open_viewer()

    def _init_ids(self) -> None:
        self.goal_site_id = self._name_to_id(mujoco.mjtObj.mjOBJ_SITE, "goal_region")
        self.ee_target_site_id = self._name_to_id(mujoco.mjtObj.mjOBJ_SITE, "ee_target_site")
        self.target_body_id = self._name_to_id(mujoco.mjtObj.mjOBJ_BODY, "target_cube")
        self.distractor_body_id = self._name_to_id(mujoco.mjtObj.mjOBJ_BODY, "distractor_cube")
        self.target_joint_id = self._name_to_id(mujoco.mjtObj.mjOBJ_JOINT, "target_cube_joint")
        self.distractor_joint_id = self._name_to_id(mujoco.mjtObj.mjOBJ_JOINT, "distractor_cube_joint")
        self.floor_geom_id = self._name_to_id(mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.finger_joint1_id = self._name_to_id(mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
        self.finger_joint2_id = self._name_to_id(mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")

    def _open_viewer(self) -> None:
        try:
            import mujoco.viewer
        except Exception as exc:  # pragma: no cover - render path is not used in CI tests
            self.logger.log_event("viewer_unavailable", {"error": repr(exc)})
            self.viewer = None
            return
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def _name_to_id(self, obj_type: mujoco.mjtObj, name: str) -> int:
        obj_id = mujoco.mj_name2id(self.model, obj_type, name)
        if obj_id < 0:
            raise ValueError(f"Missing {obj_type.name} named '{name}' in model.")
        return int(obj_id)

    def _set_freejoint_pose(self, joint_id: int, pos_xyz: np.ndarray) -> None:
        qpos_adr = int(self.model.jnt_qposadr[joint_id])
        qvel_adr = int(self.model.jnt_dofadr[joint_id])
        self.data.qpos[qpos_adr : qpos_adr + 3] = pos_xyz
        self.data.qpos[qpos_adr + 3 : qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.data.qvel[qvel_adr : qvel_adr + 6] = 0.0

    def _set_robot_home(self) -> None:
        home_q = np.array([0.0, 0.3, 0.0, -1.57079, 0.0, 2.0, -0.7853], dtype=float)
        for idx, qpos_idx in enumerate(self.controller.arm_qpos_ids):
            self.data.qpos[qpos_idx] = home_q[idx]
        finger1_qpos = int(self.model.jnt_qposadr[self.finger_joint1_id])
        finger2_qpos = int(self.model.jnt_qposadr[self.finger_joint2_id])
        self.data.qpos[finger1_qpos] = self.controller.config.gripper_open_value
        self.data.qpos[finger2_qpos] = self.controller.config.gripper_open_value
        for ctrl_idx, q in zip(self.controller.arm_ctrl_ids, home_q):
            self.data.ctrl[ctrl_idx] = q
        self.data.ctrl[self.controller.gripper_ctrl_id] = self.controller.config.gripper_open_value
        if hasattr(self.data, "eq_active"):
            self.data.eq_active[self.controller.weld_id] = 0
        self.controller.grasp_attached = False
        self.controller.contact_persistence_steps = 0

    def _sample_xy(self, existing: list[np.ndarray], *, min_dist: float = 0.10) -> np.ndarray:
        x_low, y_low = self.config.workspace_low[:2]
        x_high, y_high = self.config.workspace_high[:2]
        for _ in range(200):
            candidate = np.array(
                [self.rng.uniform(x_low + 0.05, x_high - 0.05), self.rng.uniform(y_low + 0.05, y_high - 0.05)],
                dtype=float,
            )
            if all(np.linalg.norm(candidate - prev) >= min_dist for prev in existing):
                return candidate
        raise RuntimeError("Could not sample non-overlapping XY positions.")

    def _randomize_layout(self) -> dict[str, np.ndarray]:
        sampled_xy: list[np.ndarray] = []
        target_xy = self._sample_xy(sampled_xy, min_dist=0.12)
        sampled_xy.append(target_xy)
        distractor_xy = self._sample_xy(sampled_xy, min_dist=0.12)
        sampled_xy.append(distractor_xy)
        goal_xy = self._sample_xy(sampled_xy, min_dist=0.12)
        target_pos = np.array([target_xy[0], target_xy[1], 0.03], dtype=float)
        distractor_pos = np.array([distractor_xy[0], distractor_xy[1], 0.03], dtype=float)
        goal_pos = np.array([goal_xy[0], goal_xy[1], 0.001], dtype=float)
        self._set_freejoint_pose(self.target_joint_id, target_pos)
        self._set_freejoint_pose(self.distractor_joint_id, distractor_pos)
        self.model.site_pos[self.goal_site_id] = goal_pos
        return {
            "target_pos": target_pos,
            "distractor_pos": distractor_pos,
            "goal_pos": goal_pos,
        }

    def reset(self, *, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self._set_robot_home()
        layout = self._randomize_layout()
        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0
        self.last_action = np.zeros(4, dtype=float)
        self.current_ee_target = np.array(self.data.site_xpos[self.controller.ee_site_id], dtype=float)
        self.reference_ee_quat = self.controller.current_ee_quat()
        self.model.site_pos[self.ee_target_site_id] = self.current_ee_target
        obs = self._get_observation(self.last_action)
        self.logger.log_event("reset", layout)
        return obs

    def _clip_workspace(self, xyz: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(xyz, dtype=float), self.config.workspace_low, self.config.workspace_high)

    def _rate_limit_ee_target(self, desired_xyz: np.ndarray) -> np.ndarray:
        desired_xyz = self._clip_workspace(desired_xyz)
        delta = desired_xyz - self.current_ee_target
        delta_norm = float(np.linalg.norm(delta))
        max_delta = float(self.config.ee_target_max_delta)
        if delta_norm > max_delta > 0.0:
            delta = delta * (max_delta / delta_norm)
        return self._clip_workspace(self.current_ee_target + delta)

    def _body_pos(self, body_id: int) -> np.ndarray:
        return np.array(self.data.xpos[body_id], dtype=float)

    def _gripper_state(self) -> dict[str, Any]:
        finger1_qpos = int(self.model.jnt_qposadr[self.finger_joint1_id])
        finger2_qpos = int(self.model.jnt_qposadr[self.finger_joint2_id])
        width = float(self.data.qpos[finger1_qpos] + self.data.qpos[finger2_qpos])
        return {
            "width": width,
            "closed": bool(width < 0.02),
            "attached": bool(self.controller.grasp_attached),
        }

    def _contacts_summary(self) -> dict[str, Any]:
        target_hand = False
        target_floor = False
        pairs: list[tuple[str, str]] = []
        for idx in range(int(self.data.ncon)):
            contact = self.data.contact[idx]
            g1 = int(contact.geom1)
            g2 = int(contact.geom2)
            n1 = _name_or_fallback(self.model, mujoco.mjtObj.mjOBJ_GEOM, g1)
            n2 = _name_or_fallback(self.model, mujoco.mjtObj.mjOBJ_GEOM, g2)
            if len(pairs) < 8:
                pairs.append((n1, n2))
            if self.controller.target_geom_id in (g1, g2):
                other = g2 if g1 == self.controller.target_geom_id else g1
                if other in self.controller.hand_geom_ids:
                    target_hand = True
                if other == self.floor_geom_id:
                    target_floor = True
        return {
            "ncon": int(self.data.ncon),
            "target_hand_contact": target_hand,
            "target_floor_contact": target_floor,
            "pairs": pairs,
        }

    def _get_observation(self, action: np.ndarray) -> dict[str, Any]:
        target_pos = self._body_pos(self.target_body_id)
        distractor_pos = self._body_pos(self.distractor_body_id)
        goal_pos = np.array(self.data.site_xpos[self.goal_site_id], dtype=float)
        obs: dict[str, Any] = {
            "qpos": np.array(self.data.qpos, dtype=float),
            "qvel": np.array(self.data.qvel, dtype=float),
            "ee_pos": np.array(self.data.site_xpos[self.controller.ee_site_id], dtype=float),
            "target_pos": target_pos,
            "distractor_pos": distractor_pos,
            "goal_pos": goal_pos,
            "gripper_state": self._gripper_state(),
            "action": np.array(action, dtype=float),
            "contacts_summary": self._contacts_summary(),
        }
        return obs

    def _is_success(self, obs: dict[str, Any]) -> bool:
        target_pos = np.asarray(obs["target_pos"], dtype=float)
        goal_pos = np.asarray(obs["goal_pos"], dtype=float)
        target_goal_dist = float(np.linalg.norm(target_pos[:2] - goal_pos[:2]))
        placed_height_ok = bool(target_pos[2] < 0.07)
        released_cmd = bool(self.last_action[3] <= 0.5)
        released_state = bool((not obs["gripper_state"]["attached"]) and (obs["gripper_state"]["width"] > 0.008))
        released = bool(released_cmd and released_state)
        return bool(target_goal_dist < 0.06 and placed_height_ok and released)

    def _run_sanity_checks(self, obs: dict[str, Any]) -> tuple[bool, list[str]]:
        errors: list[str] = []
        qpos = np.asarray(obs["qpos"], dtype=float)
        qvel = np.asarray(obs["qvel"], dtype=float)
        ee_pos = np.asarray(obs["ee_pos"], dtype=float)
        target_pos = np.asarray(obs["target_pos"], dtype=float)
        goal_pos = np.asarray(obs["goal_pos"], dtype=float)

        if not np.isfinite(qpos).all():
            errors.append("Non-finite qpos detected.")
        if not np.isfinite(qvel).all():
            errors.append("Non-finite qvel detected.")
        if not np.isfinite(ee_pos).all():
            errors.append("Non-finite ee_pos detected.")
        if float(np.max(np.abs(qvel))) > 80.0:
            errors.append("Joint velocity exceeded 80 rad/s-equivalent limit.")
        if ee_pos[2] < 0.0:
            errors.append("End-effector went below floor plane (z < 0).")
        if self.controller.joint_limit_min_margin() < 0.004:
            errors.append("Arm joint reached near-limit unsafe margin (<0.004 rad).")
        if target_pos[2] < -0.01:
            errors.append("Target cube dropped below floor.")
        if float(np.linalg.norm(target_pos - goal_pos)) > 1.5:
            errors.append("Target-goal distance exploded (>1.5m).")
        if bool(obs["gripper_state"]["attached"]) and float(np.linalg.norm(ee_pos - target_pos)) > 0.25:
            errors.append("Grasp weld active while EE and target are too far apart.")

        ok = len(errors) == 0
        if not ok:
            self.logger.log_event("sanity_violation", {"errors": errors}, step=self.step_count)
        return ok, errors

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=float).reshape(-1)
        if action.shape[0] != 4:
            raise ValueError("Action must be shape (4,) -> [ee_x, ee_y, ee_z, gripper_closed_flag].")

        self.last_action = action.copy()
        self.current_ee_target = self._rate_limit_ee_target(action[:3])
        gripper_closed = bool(action[3] > 0.5)
        self.model.site_pos[self.ee_target_site_id] = self.current_ee_target
        # Position-only control for robustness in this baseline.
        # Orientation lock can over-constrain IK for some seeds/pick poses.
        use_orient_lock = False
        ee_target_quat = self.reference_ee_quat if use_orient_lock else None

        attachment_changed = False
        for _ in range(self.config.frame_skip):
            attached_before = self.controller.grasp_attached
            ctrl_info = self.controller.apply(
                self.current_ee_target,
                gripper_closed=gripper_closed,
                ee_target_quat=ee_target_quat,
            )
            mujoco.mj_step(self.model, self.data)
            if ctrl_info["grasp_attached"] != attached_before:
                attachment_changed = True
            if self.viewer is not None:
                self.viewer.sync()

        self.step_count += 1
        obs = self._get_observation(action)
        success = self._is_success(obs)
        done = bool(success or self.step_count >= self.config.max_steps)
        reward = 1.0 if success else 0.0
        sanity_ok, sanity_errors = self._run_sanity_checks(obs)
        info = {
            "success": success,
            "sanity_ok": sanity_ok,
            "sanity_errors": sanity_errors,
            "grasp_attached": bool(self.controller.grasp_attached),
            "attachment_changed": attachment_changed,
            "joint_limit_min_margin": self.controller.joint_limit_min_margin(),
        }
        if info["joint_limit_min_margin"] < 0.03:
            self.logger.log_event(
                "joint_limit_warning",
                {"joint_limit_min_margin": info["joint_limit_min_margin"]},
                step=self.step_count,
            )
        if attachment_changed:
            self.logger.log_event(
                "grasp_state_changed",
                {
                    "attached": bool(self.controller.grasp_attached),
                    "target_hand_contact": obs["contacts_summary"]["target_hand_contact"],
                    "gripper_closed_cmd": gripper_closed,
                    "contact_persistence_steps": ctrl_info["contact_persistence_steps"],
                    "target_rel_speed_ok": ctrl_info["target_rel_speed_ok"],
                },
                step=self.step_count,
            )
        self.logger.log_step(self.step_count, obs, info)
        if self.config.sanity_asserts and not sanity_ok:
            raise AssertionError(f"Sanity checks failed at step {self.step_count}: {sanity_errors}")
        return obs, reward, done, info

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        self.logger.close()

    def __enter__(self) -> "PandaPickPlaceEnv":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke runner for Panda pick-and-place environment.")
    render_group = parser.add_mutually_exclusive_group()
    render_group.add_argument("--render", action="store_true", help="Enable MuJoCo viewer.")
    render_group.add_argument("--headless", action="store_true", help="Force headless mode.")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/smoke"))
    parser.add_argument(
        "--no-sanity-asserts",
        action="store_true",
        help="Do not raise on sanity violations (still logged).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    render_mode = bool(args.render and not args.headless)
    config = EnvConfig(
        seed=args.seed,
        render=render_mode,
        output_dir=args.output_dir,
        sanity_asserts=not args.no_sanity_asserts,
    )
    with PandaPickPlaceEnv(config) as env:
        obs = env.reset(seed=args.seed)
        start_ee = np.array(obs["ee_pos"], dtype=float)
        phase = "approach"
        retry_count = 0
        close_hold_steps = 0
        open_hold_steps = 0
        max_pick_retries = 3

        def transition(next_phase: str, step_idx: int) -> None:
            nonlocal phase, close_hold_steps, open_hold_steps
            if next_phase == phase:
                return
            phase = next_phase
            close_hold_steps = 0
            open_hold_steps = 0
            env.logger.log_event("phase_transition", {"phase": phase}, step=step_idx)

        for t in range(args.steps):
            target = np.array(obs["target_pos"], dtype=float)
            goal = np.array(obs["goal_pos"], dtype=float)
            ee = np.array(obs["ee_pos"], dtype=float)
            attached = bool(obs["gripper_state"]["attached"])
            target_on_floor = bool(target[2] < 0.045)

            pre_grasp = np.array([target[0], target[1], max(target[2] + 0.22, 0.22)], dtype=float)
            grasp = np.array([target[0], target[1], max(target[2] + 0.075, 0.085)], dtype=float)
            lift = np.array([start_ee[0], start_ee[1], 0.30], dtype=float)
            pre_place = np.array([goal[0], goal[1], 0.23], dtype=float)
            place = np.array([goal[0], goal[1], 0.10], dtype=float)

            xy_err_target = float(np.linalg.norm(ee[:2] - target[:2]))
            dist_to_pre_grasp = float(np.linalg.norm(ee - pre_grasp))
            dist_to_grasp = float(np.linalg.norm(ee - grasp))
            target_goal_xy = float(np.linalg.norm(target[:2] - goal[:2]))

            if phase == "approach":
                ee_cmd = pre_grasp
                gripper = 0.0
                if dist_to_pre_grasp < 0.06:
                    transition("descend", t + 1)
            elif phase == "descend":
                ee_cmd = grasp
                gripper = 0.0
                if dist_to_grasp < 0.045 or obs["contacts_summary"]["target_hand_contact"]:
                    transition("close", t + 1)
            elif phase == "close":
                ee_cmd = grasp
                gripper = 1.0
                close_hold_steps += 1
                if attached:
                    transition("lift", t + 1)
                elif close_hold_steps > 28:
                    retry_count += 1
                    if retry_count > max_pick_retries:
                        transition("abort", t + 1)
                    else:
                        transition("approach", t + 1)
            elif phase == "lift":
                ee_cmd = lift
                gripper = 1.0
                if not attached and target_on_floor:
                    transition("approach", t + 1)
                elif target[2] > 0.06:
                    transition("move_to_goal", t + 1)
            elif phase == "move_to_goal":
                ee_cmd = pre_place
                gripper = 1.0
                if not attached and target_on_floor:
                    transition("approach", t + 1)
                elif target_goal_xy < 0.08:
                    transition("lower_to_place", t + 1)
            elif phase == "lower_to_place":
                ee_cmd = place
                gripper = 1.0
                if ee[2] < 0.12:
                    transition("open", t + 1)
            elif phase == "open":
                ee_cmd = place
                gripper = 0.0
                open_hold_steps += 1
                if open_hold_steps > 18:
                    transition("retreat", t + 1)
            elif phase == "retreat":
                ee_cmd = pre_place
                gripper = 0.0
            else:
                ee_cmd = np.array([start_ee[0], start_ee[1], 0.24], dtype=float)
                gripper = 0.0

            action = np.array([ee_cmd[0], ee_cmd[1], ee_cmd[2], gripper], dtype=float)
            obs, reward, done, info = env.step(action)
            if phase == "abort":
                env.logger.log_event(
                    "episode_abort",
                    {"reason": "max_pick_retries_exceeded", "retry_count": retry_count},
                    step=t + 1,
                )
                done = True
            if done:
                env.logger.log_event(
                    "episode_done",
                    {"step": t + 1, "reward": reward, "info": info, "phase": phase, "retry_count": retry_count},
                )
                break


if __name__ == "__main__":
    main()
