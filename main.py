# main.py
import numpy as np
import mujoco
import mujoco.viewer

from env import PickPlaceEnv
from controller import JacobianIKController


def main():
    env = PickPlaceEnv("model.xml", hz=50)
    ctrl = JacobianIKController(env.model)

    env.reset(seed=0)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            # stato
            state = env.get_state(action=np.zeros(env.nu))
            cube = state["object_pos"]

            # target: sopra il cubo (solo smoke test)
            target = cube.copy()
            target[2] = 0.12

            q_des = ctrl.solve_qpos_step(env.data, target)

            action = np.zeros(env.nu)
            action[:4] = q_des                 # <-- ora è target posizione
            action[4] = ctrl.gripper_cmd(open_=True)

            env.step(action)
            viewer.sync()


if __name__ == "__main__":
    main()