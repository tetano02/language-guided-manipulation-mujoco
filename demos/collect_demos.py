import numpy as np
import mujoco
import mujoco.viewer

from env import PickPlaceEnv
from controller import JacobianIKController

def main():
    env = PickPlaceEnv("mujoco_env/model.xml", hz=50)
    ctrl = JacobianIKController(env.model)

    env.reset(seed=0)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for t in range(2000):
            state = env.get_state(action=np.zeros(env.nu))
            ee = state["ee_pos"]
            cube = state["object_pos"]

            # trivial: hover above cube (no grasp yet)
            target = cube.copy()
            target[2] = 0.12

            q_des = ctrl.solve_qpos_step(env.data, target)
            u_arm = ctrl.arm_pd_action(env.data, q_des)
            u_grip = ctrl.gripper_cmd(open_=True)

            action = np.zeros(env.nu)
            action[:4] = u_arm
            action[4] = u_grip

            env.step(action)
            viewer.sync()

if __name__ == "__main__":
    main()