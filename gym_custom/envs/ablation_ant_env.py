# 2021/4/28
# ant-v2の終了条件の意味や有効性を検証する環境
# for ablation study
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import mujoco_py
import math

class AblationAntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    quat_current = np.zeros(4)
    vec = np.zeros(3)
    
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'ant.xml', 5)
        utils.EzPickle.__init__(self)
        self.quat_current = np.array([1.0, 0.0, 0.0, 0.0])
        self.vec = np.array([0.0, 0.0, 1.0])

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        body_posbefore = np.array([self.get_body_com("torso")[0],self.get_body_com("torso")[1],self.get_body_com("torso")[2]])
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        body_posafter = np.array([self.get_body_com("torso")[0],self.get_body_com("torso")[1],self.get_body_com("torso")[2]])
        quat_after = self.data.get_body_xquat("torso")

        forward_reward = (xposafter - xposbefore) / self.dt
        ctrl_cost = 0.5 * np.square(a).sum()
        contact_cost = (
            0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        )
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()

        # 終了条件の記述
        # before
        # notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        # after
        # notdone = np.isfinite(state).all() and state[2] <= 1.0
        # notdone = np.isfinite(state).all() and state[2] >= 0.2 
        # notdone = np.isfinite(state).all()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.5

        res = np.zeros(4)
        mujoco_py.functions.mju_mulQuat(res, self.quat_current, quat_after)
        if res[0] < 0:
            res = res * -1

        torso_vec = np.zeros(3)
        mujoco_py.functions.mju_rotVecQuat(torso_vec, self.vec, quat_after)
        # Terminate the episode when the agent falls over.
        if torso_vec[2] < -0.8:
            notdone = False

        self.quat_current = res

        done = not notdone
        ob = self._get_obs()
        return (
            ob,
            reward,
            done,
            dict(
                reward_forward=forward_reward,
                reward_ctrl=-ctrl_cost,
                reward_contact=-contact_cost,
                reward_survive=survive_reward,
            ),
        )

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat,
                np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ]
        )

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.1, high=0.1
        )
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.05
        self.viewer.cam.elevation = -90