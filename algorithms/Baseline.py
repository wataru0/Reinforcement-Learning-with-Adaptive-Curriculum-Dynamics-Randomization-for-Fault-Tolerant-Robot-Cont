# Baseline RL Algorithm

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
            joint_mask = [i for i,x in enumerate(self.cripple_mask) if x == 99]
            if joint_mask != []:
                action[joint_mask[0]] = action[joint_mask[0]] * self.k
                action[joint_mask[1]] = action[joint_mask[1]] * self.k

        obs, reward, done, info = self.env.step(action)

        self.cReward += reward
        
        return obs, reward, done, info

    def reset_task(self,value=None):
        
        self.crippled_leg = value if value is not None else np.random.randint(0,5)

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
