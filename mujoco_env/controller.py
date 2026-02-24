"""EE-first controller with damped least-squares IK and simplified grasp weld logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import mujoco
import numpy as np


ARM_JOINT_NAMES: tuple[str, ...] = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
)


@dataclass
class EEControllerConfig:
    damping: float = 1e-3
    ik_step_size: float = 0.35
    max_delta_q: float = 0.08
    use_orientation: bool = False
    orientation_weight: float = 0.2
    gripper_open_value: float = 0.04
    gripper_close_value: float = 0.0


class EEController:
    """End-effector target controller over Panda's 7 arm joints."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        config: EEControllerConfig | None = None,
        *,
        ee_site_name: str = "gripper",
    ) -> None:
        self.model = model
        self.data = data
        self.config = config or EEControllerConfig()
        self.ee_site_id = self._name_to_id(mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
        self.arm_joint_ids = [self._name_to_id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINT_NAMES]
        self.arm_qpos_ids = [int(self.model.jnt_qposadr[jid]) for jid in self.arm_joint_ids]
        self.arm_dof_ids = [int(self.model.jnt_dofadr[jid]) for jid in self.arm_joint_ids]
        self.arm_ctrl_ids = [
            self._name_to_id(mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i}") for i in range(1, 8)
        ]
        self.gripper_ctrl_id = self._name_to_id(mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
        self.target_geom_id = self._name_to_id(mujoco.mjtObj.mjOBJ_GEOM, "target_cube_geom")
        self.weld_id = self._name_to_id(mujoco.mjtObj.mjOBJ_EQUALITY, "target_grasp_weld")
        self.hand_body_id = self._name_to_id(mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.target_body_id = self._name_to_id(mujoco.mjtObj.mjOBJ_BODY, "target_cube")
        self.hand_geom_ids = self._collect_hand_geom_ids(("hand", "left_finger", "right_finger"))
        self.grasp_geom_ids = self._collect_hand_geom_ids(("left_finger", "right_finger"))
        for pad_name in ("left_finger_pad", "right_finger_pad"):
            pad_id = self._try_name_to_id(mujoco.mjtObj.mjOBJ_GEOM, pad_name)
            if pad_id is not None:
                self.grasp_geom_ids.add(pad_id)
        self.grasp_attached = False

    def _name_to_id(self, obj_type: mujoco.mjtObj, name: str) -> int:
        obj_id = mujoco.mj_name2id(self.model, obj_type, name)
        if obj_id < 0:
            raise ValueError(f"Missing {obj_type.name} named '{name}' in model.")
        return int(obj_id)

    def _try_name_to_id(self, obj_type: mujoco.mjtObj, name: str) -> int | None:
        obj_id = mujoco.mj_name2id(self.model, obj_type, name)
        if obj_id < 0:
            return None
        return int(obj_id)

    def _collect_hand_geom_ids(self, body_names: Iterable[str]) -> set[int]:
        body_ids = {
            self._name_to_id(mujoco.mjtObj.mjOBJ_BODY, name)
            for name in body_names
        }
        geom_ids: set[int] = set()
        for geom_id in range(self.model.ngeom):
            if int(self.model.geom_bodyid[geom_id]) in body_ids:
                geom_ids.add(geom_id)
        return geom_ids

    def current_ee_pos(self) -> np.ndarray:
        return np.array(self.data.site_xpos[self.ee_site_id], dtype=float)

    def compute_qpos_des(
        self,
        ee_target_pos: np.ndarray,
        ee_target_quat: np.ndarray | None = None,
    ) -> np.ndarray:
        ee_target_pos = np.asarray(ee_target_pos, dtype=float).reshape(3)
        pos_error = ee_target_pos - self.current_ee_pos()

        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
        jac_arm = jacp[:, self.arm_dof_ids]
        err_vec = pos_error

        if self.config.use_orientation and ee_target_quat is not None:
            rot_error = self._orientation_error(np.asarray(ee_target_quat, dtype=float).reshape(4))
            jac_arm = np.vstack((jac_arm, self.config.orientation_weight * jacr[:, self.arm_dof_ids]))
            err_vec = np.concatenate((pos_error, self.config.orientation_weight * rot_error))

        damping_sq = self.config.damping * self.config.damping
        gram = jac_arm @ jac_arm.T + damping_sq * np.eye(jac_arm.shape[0], dtype=float)
        dq = jac_arm.T @ np.linalg.solve(gram, err_vec)

        dq = np.clip(dq, -self.config.max_delta_q, self.config.max_delta_q)
        q_current = np.array([self.data.qpos[idx] for idx in self.arm_qpos_ids], dtype=float)
        q_des = q_current + self.config.ik_step_size * dq

        for i, joint_id in enumerate(self.arm_joint_ids):
            q_min = float(self.model.jnt_range[joint_id, 0])
            q_max = float(self.model.jnt_range[joint_id, 1])
            q_des[i] = np.clip(q_des[i], q_min, q_max)

        return q_des

    def _orientation_error(self, target_quat: np.ndarray) -> np.ndarray:
        xmat = np.array(self.data.site_xmat[self.ee_site_id], dtype=float).reshape(9)
        current_quat = np.zeros(4, dtype=float)
        mujoco.mju_mat2Quat(current_quat, xmat)
        inv_current = current_quat.copy()
        inv_current[1:] *= -1.0
        q_err = np.zeros(4, dtype=float)
        mujoco.mju_mulQuat(q_err, target_quat, inv_current)
        if q_err[0] < 0.0:
            q_err *= -1.0
        return 2.0 * q_err[1:]

    def target_in_hand_contact(self) -> bool:
        for idx in range(int(self.data.ncon)):
            contact = self.data.contact[idx]
            geom_1 = int(contact.geom1)
            geom_2 = int(contact.geom2)
            if self.target_geom_id not in (geom_1, geom_2):
                continue
            other = geom_2 if geom_1 == self.target_geom_id else geom_1
            if other in self.grasp_geom_ids:
                return True
        return False

    def _set_weld_relative_pose(self) -> None:
        hand_pos = np.array(self.data.xpos[self.hand_body_id], dtype=float)
        target_pos = np.array(self.data.xpos[self.target_body_id], dtype=float)
        hand_rot = np.array(self.data.xmat[self.hand_body_id], dtype=float).reshape(3, 3)
        rel_pos = hand_rot.T @ (target_pos - hand_pos)

        hand_quat = np.array(self.data.xquat[self.hand_body_id], dtype=float)
        target_quat = np.array(self.data.xquat[self.target_body_id], dtype=float)
        inv_hand_quat = hand_quat.copy()
        inv_hand_quat[1:] *= -1.0
        rel_quat = np.zeros(4, dtype=float)
        mujoco.mju_mulQuat(rel_quat, inv_hand_quat, target_quat)
        rel_quat /= np.linalg.norm(rel_quat) + 1e-12

        # MuJoCo weld eq_data layout:
        # [0:3]=anchor, [3:6]=rel_pos, [6:10]=rel_quat (wxyz), [10]=torquescale.
        self.model.eq_data[self.weld_id, 0:3] = 0.0
        self.model.eq_data[self.weld_id, 3:6] = rel_pos
        self.model.eq_data[self.weld_id, 6:10] = rel_quat

    def _set_grasp_weld(self, active: bool) -> None:
        if not hasattr(self.data, "eq_active"):
            return
        if active:
            # Prevent impulsive jumps by welding with the current relative hand-target transform.
            self._set_weld_relative_pose()
        self.data.eq_active[self.weld_id] = 1 if active else 0
        mujoco.mj_forward(self.model, self.data)
        self.grasp_attached = active

    def apply(
        self,
        ee_target_pos: np.ndarray,
        gripper_closed: bool,
        ee_target_quat: np.ndarray | None = None,
    ) -> dict[str, bool]:
        q_des = self.compute_qpos_des(ee_target_pos, ee_target_quat=ee_target_quat)
        for ctrl_idx, q_idx in zip(self.arm_ctrl_ids, range(len(q_des))):
            self.data.ctrl[ctrl_idx] = q_des[q_idx]
        self.data.ctrl[self.gripper_ctrl_id] = (
            self.config.gripper_close_value if gripper_closed else self.config.gripper_open_value
        )

        target_contact = self.target_in_hand_contact()
        ee_to_target = float(np.linalg.norm(self.current_ee_pos() - np.array(self.data.xpos[self.target_body_id])))
        within_grasp_distance = ee_to_target < 0.10

        if gripper_closed and target_contact and within_grasp_distance and not self.grasp_attached:
            self._set_grasp_weld(True)
        elif (not gripper_closed) and self.grasp_attached:
            self._set_grasp_weld(False)
        elif self.grasp_attached and ee_to_target > 0.30:
            # Failsafe: do not keep a stale weld active if simulation diverges.
            self._set_grasp_weld(False)

        return {
            "target_contact": target_contact,
            "grasp_attached": self.grasp_attached,
        }
