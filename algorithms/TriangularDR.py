# Triangular Domain Randomization Algorithm
import gym
import numpy as np
import random

rewardlist = []

class TDREnv(gym.Wrapper):
    def __init__(self,env,value=None):
        super().__init__(env) 
        # crippled leg number
        self.value = value 
        self.crippled_leg = 0
        self.cripple_mask = np.ones(self.action_space.shape)
        self._init_geom_rgba = self.model.geom_rgba.copy()
        self.cReward = 0
        self.joint_range = 1.0

    def reset(self,**kwargs):
        self.reset_task()
        rewardlist.append(self.cReward)
        self.cReward = 0 
        return self.env.reset(**kwargs)

    def step(self,action):
        if self.cripple_mask is not None:
            joint_mask = [i for i,x in enumerate(self.cripple_mask) if x == 99]
            if joint_mask != []:
                action[joint_mask[0]]=henkan(action[joint_mask[0]],-1,1, -self.joint_range, self.joint_range)
                action[joint_mask[1]]=henkan(action[joint_mask[1]],-1,1, -self.joint_range, self.joint_range)
            #ーーーー action = self.cripple_mask * action
            

        obs, reward, done, info = self.env.step(action)

        self.cReward += reward
        
        return obs, reward, done, info

    def reset_task(self,value=None):
        # randomly cripple leg (4 is nothing)
        self.crippled_leg = value if value is not None else np.random.randint(0,5)
        
        self.joint_range = random.triangular(low=0.0, high=1.5, mode=1.5)
        
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

def henkan(val,start1,stop1,start2,stop2):
    return start2 + (stop2 - start2) * ((val-start1)/(stop1 - start1))