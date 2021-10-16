# Baseline RL Algorithm
# 2021/07/09
# 今まで一つのファイルでトレーニングを回していたのをクラスごとにしっかり分けようというもの
# kの値をtrain開始時に指定できるように改良
# train.pyを実行する際に，gym環境をラッパーすることで実装する


import gym
import numpy as np
import random

rewardlist = []

class BaselineEnv(gym.Wrapper):
    def __init__(self, env, value=None, k=1.0):
        super().__init__(env) 
        # crippled leg number
        self.value = value 
        self.crippled_leg = 0
        self.cripple_mask = np.ones(self.action_space.shape)
        self._init_geom_rgba = self.model.geom_rgba.copy()
        self.cReward = 0
        self.k = k

    def reset(self,**kwargs):
        self.reset_task()
        rewardlist.append(self.cReward)
        self.cReward = 0 
        return self.env.reset(**kwargs)

    def step(self,action):
        if self.cripple_mask is not None:
            joint_mask = [i for i,x in enumerate(self.cripple_mask) if x == 99] # 99が入っているインデックスを取得
            #print(joint_mask) # [4,5]のように表示される
            if joint_mask != []:
                action[joint_mask[0]] = action[joint_mask[0]] * self.k
                action[joint_mask[1]] = action[joint_mask[1]] * self.k

        obs, reward, done, info = self.env.step(action)

        self.cReward += reward
        
        return obs, reward, done, info

    def reset_task(self,value=None):
        # randomly cripple leg (4 is nothing)
        self.crippled_leg = value if value is not None else np.random.randint(0,5) # (0,4)だと0から4個なので0,1,2,3までしかでない！！

        # Pick which actuators to disable
        # joint rangeを変更する足をマスクで表現、99を代入しておく
        self.cripple_mask = np.ones(self.action_space.shape)
        if self.crippled_leg == 0:
            self.cripple_mask[2] = 99
            self.cripple_mask[3] = 99
        elif self.crippled_leg == 1:
            self.cripple_mask[4] = 99
            self.cripple_mask[5] = 99
        elif self.crippled_leg == 2:
            self.cripple_mask[6] = 99
            self.cripple_mask[7] = 99
        elif self.crippled_leg == 3:
            self.cripple_mask[0] = 99
            self.cripple_mask[1] = 99
        elif self.crippled_leg == 4:
            pass

        # make th removed leg look red
        geom_rgba = self._init_geom_rgba.copy()
        if self.crippled_leg == 0:
            geom_rgba[3, :3] = np.array([1, 0, 0])
            geom_rgba[4, :3] = np.array([1, 0, 0])
        elif self.crippled_leg == 1:
            geom_rgba[6, :3] = np.array([1, 0, 0])
            geom_rgba[7, :3] = np.array([1, 0, 0])
        elif self.crippled_leg == 2:
            geom_rgba[9, :3] = np.array([1, 0, 0])
            geom_rgba[10, :3] = np.array([1, 0, 0])
        elif self.crippled_leg == 3:
            geom_rgba[12, :3] = np.array([1, 0, 0])
            geom_rgba[13, :3] = np.array([1, 0, 0])
        self.model.geom_rgba[:] = geom_rgba 
