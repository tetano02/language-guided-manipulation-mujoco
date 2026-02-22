import numpy as np
import mujoco

class PickPlaceEnv:
    def __init__(self, model_path: str, hz: int = 50):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.hz = hz
        self.dt = self.model.opt.timestep
        self.substeps = max(1, int(round((1.0 / hz) / self.dt)))

        # Cache ids (veloce e pulito)
        self.site_ee = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        self.site_cube = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "cube_site")
        self.site_goal = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "goal_site")
        self.j_grip = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "grip")

        # actuators: a1 a2 a3 a4 grip_pos
        self.nu = self.model.nu

    def reset(self, seed: int | None = None):
        if seed is not None:
            np.random.seed(seed)

        mujoco.mj_resetData(self.model, self.data)

        # posa comoda (leggermente verso il cubo)
        init = np.array([0.0, 0.8, -0.8, 0.0])
        for i, jn in enumerate(["j1", "j2", "j3", "j4"]):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            qadr = self.model.jnt_qposadr[jid]
            self.data.qpos[qadr] = init[i]

        # Randomize cube and goal (planar)
        # cube body is freejoint => qpos includes 7 values at the end for that joint
        # easiest: set cube body pos directly via body_xpos? nope (computed).
        # We'll set freejoint qpos: [x, y, z, qw, qx, qy, qz]
        cube_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube")
        # If name-to-id doesn't find (because freejoint unnamed), fallback:
        # We'll instead find cube body id and map to its joint address.
        cube_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        jadr = self.model.body_jntadr[cube_bid]
        qadr = self.model.jnt_qposadr[jadr]

        x = np.random.uniform(0.25, 0.45)
        y = np.random.uniform(-0.15, 0.15)
        z = 0.03

        self.data.qpos[qadr:qadr+7] = np.array([x, y, z, 1, 0, 0, 0], dtype=float)

        # goal body has fixed pos in XML; for randomization you can:
        goal_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal")
        gx = np.random.uniform(-0.45, -0.25)
        gy = np.random.uniform(-0.15, 0.15)
        # Move the goal body (it's kinematic). This changes model, but ok for our use.
        self.model.body_pos[goal_bid] = np.array([gx, gy, 0.001])

        mujoco.mj_forward(self.model, self.data)
        return self.get_state(action=np.zeros(self.nu))

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=float).copy()
        assert action.shape == (self.nu,)

        self.data.ctrl[:] = action

        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)

        obs = self.get_state(action=action)
        reward = 0.0
        done = False
        info = {"success": self.is_success()}
        return obs, reward, done, info

    def get_state(self, action: np.ndarray):
        ee = self.data.site_xpos[self.site_ee].copy()
        cube = self.data.site_xpos[self.site_cube].copy()
        grip = float(self.data.qpos[self.model.jnt_qposadr[self.j_grip]])

        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "ee_pos": ee,
            "object_pos": cube,
            "gripper_state": grip,   # continuous aperture
            "action": action.copy(),
        }

    def is_success(self, tol: float = 0.06):
        cube = self.data.site_xpos[self.site_cube]
        goal = self.data.site_xpos[self.site_goal]
        # planar distance + small height tolerance
        dxy = np.linalg.norm(cube[:2] - goal[:2])
        return (dxy < tol) and (cube[2] < 0.06)