# 2021/4/21
# ant-v2の終了条件を研究用に改良するためにテストする環境
# antが転倒した時の検出を行うために試行錯誤する環境

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
import mujoco_py
import math
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../utils'))
import quaternion
from quaternion import my_mulQuat

class TestAntEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    quat_current = np.zeros(4)
    vec = np.zeros(3)
    
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'ant.xml', 5)
        utils.EzPickle.__init__(self)
        # self.quat_current = np.array([math.cos(0.01/2), math.sin(0.01/2), 0.0, 0.0])
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

        # 終了条件
        notdone = np.isfinite(state).all()

        # bodyの向きをデカルト座標，クォータニオンで表現したもの=====
        # 転倒を体の向きで検知したいが，現状よくわからない
        # print("---")
        # print(self.data.body_xmat)
        # print(self.data.body_xquat)
        # print(self.data.get_body_xquat("torso"))

        res = np.zeros(4)
        mujoco_py.functions.mju_mulQuat(res, self.quat_current, quat_after)
        # mujoco_py.functions.mju_mulQuat(res, quat_after, self.quat_current)
        # print(res, my_mulQuat(quat_after, self.quat_current))
        if res[0] < 0:
            res = res * -1
        # print(res)

        res2 = np.zeros(3)
        # mujoco_py.functions.mju_rotVecQuat(res2, self.vec, res) # これ間違い
        mujoco_py.functions.mju_rotVecQuat(res2, self.vec, quat_after) # get_body_xquat は常に最初の位置からの回転を表していた．だから[0, 0, 1]にクォータニオンをかけると，001が今どこを向いているのかが取得できる！
        # if res2[0] < 0.0:
        #     res2 = res2 * -1
        # print(res2)
        self.quat_current = res
        # ==================================================

        
        # これでtorsoの位置取得できる（x,y,z）----------------
        # print(self.data.body_xpos)
        # print(self.get_body_com('torso'))
        
        # print(body_posbefore, body_posafter)
        # print(np.linalg.norm(body_posafter - body_posbefore))
        dist = np.linalg.norm(body_posafter - body_posbefore)
        # print(dist)
        # if dist < 0.0:
        #     notdone = False
        # ------------------------------------------------

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
                res2=res2,
                dist=dist,
                res=res,
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
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.elevation = -90