import numpy as np
import mujoco

class JacobianIKController:
    def __init__(self, model: mujoco.MjModel, site_name="ee_site", arm_joint_names=("j1","j2","j3","j4")):
        self.model = model
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)

        self.joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in arm_joint_names]
        self.qpos_adrs = [model.jnt_qposadr[jid] for jid in self.joint_ids]

        self.kp = 2.0          # era 6.0
        self.dq_limit = 0.05   # era 0.15
        lam = 1e-2             # era 1e-3

        # actuator order assumed: a1 a2 a3 a4 grip_pos
        self.arm_act = 4
        self.grip_act = 4

        

    def ee_pos(self, data: mujoco.MjData):
        return data.site_xpos[self.site_id].copy()

    def solve_qpos_step(self, data: mujoco.MjData, target_xyz: np.ndarray):
        # Jacobian of site wrt dofs
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, data, jacp, jacr, self.site_id)

        # We only care about planar x,y (and a bit z if you want)
        err = target_xyz - data.site_xpos[self.site_id]
        v = self.kp * err  # desired cart vel

        # Restrict to arm dofs only
        cols = []
        for jid in self.joint_ids:
            dof_adr = self.model.jnt_dofadr[jid]
            cols.append(dof_adr)

        J = jacp[:, cols]  # (3,4)
        # Damped least squares
        lam = 1e-3
        dq = J.T @ np.linalg.solve(J @ J.T + lam*np.eye(3), v)

        dq = np.clip(dq, -self.dq_limit, self.dq_limit)

        # Produce torque-ish command via simple joint-space PD on qpos target
        q = np.array([data.qpos[a] for a in self.qpos_adrs])
        q_des = q + dq

        return q_des

    def gripper_cmd(self, open_: bool):
        # ctrl is desired joint position in meters (range -0.02..0.02)
        return 0.02 if open_ else -0.015