# Adaptive Curriculum Dynamics Randomization Algorithm

import os
import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd
import datetime

#可視化用
joint_actuator_range_max = [] 
joint_actuator_range_min = [] 
actuator_map = np.zeros((10,8)) 
actuator_bunpu= [0]*10
actuator_power_map = np.zeros((10,8)) 
rewardlist = []

# 平均報酬
best_mean_reward = -np.inf 
# 更新数
n_updates = 0 

def henkan(val, start1, stop1, start2, stop2):
    return start2 + (stop2 - start2) * ((val-start1)/(stop1 - start1))

class CDREnv(gym.Wrapper):
    def __init__(self, env, value=None, version=2, bound_fix=False):
        super().__init__(env) 
        self.value = value 
        self.version = version
        self.crippled_leg = 0
        self.cripple_mask = np.ones(self.action_space.shape)
        self._init_geom_rgba = self.model.geom_rgba.copy()
        self.joint_range = 1.0
        self.num_step = 0
        self.total_reward = 0
        # rewardを保存しておくバッファー
        self.buffer = [] 
        # default: 100
        self.buffer_size = 50
        # 前回の分布での報酬の平均値を格納しておく変数
        self.before_average = 0 
        self.joint_num = 0
        # 累積報酬
        self.cReward = 0 
        # 各関節の故障率を格納するリスト,action_spaceと同じ大きさで1に初期化
        self.actuator_list = np.ones(self.action_space.shape)
        # kのアップデートサイズ
        self.update_k_step_size = 0.01

        # minもmaxも1からスタート(v1),minもmaxも0からスタート(v2)
        if version == 1:
            # 〜6/9
            # self.joint_min = 1.0 
            # self.joint_max = 1.0 

            # 6/9〜
            self.joint_min = 1.5
            self.joint_max = 1.5 
        elif version == 2:
            self.joint_min = 0.0 
            self.joint_max = 0.0

        # Trueの時，v1の時は上限固定，v2の時は下限固定する
        self.bound_fix = bound_fix

    def reset(self, **kwargs): 
        self.reset_task()
        rewardlist.append(self.cReward)
        self.cReward = 0 

        # bufferにパフォーマンスを格納していく，buffer_sizeを超えたら評価する
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(self.total_reward)


        else: # 能力の評価
            ave = sum(self.buffer)/len(self.buffer)
            # 前より能力アップしていたら
            if self.before_average < ave: 

                # Curriculum2-v1
                if self.version == 1:
                    if self.joint_min > 0.0:
                        self.joint_min -= self.update_k_step_size
                        # kのminとmaxの差が0.1以上になったらmaxも減らす (upperfixではコメントアウト)
                        if self.bound_fix is not True:
                            if abs(self.joint_max - self.joint_min) >= 0.1:
                                self.joint_max -= self.update_k_step_size

                # Curriculum2-v2
                if self.version == 2:
                    # 〜6/9
                    # if self.joint_max <= 1.0:

                    # 6/9〜
                    if self.joint_max <= 1.5:
                        self.joint_max += self.update_k_step_size
                        # kのminとmaxの差が0.1以上になったらminも上昇 (lowerfixではコメントアウト)
                        if self.bound_fix is not True:
                            if abs(self.joint_max - self.joint_min) >= 0.1:
                                self.joint_min += self.update_k_step_size

            # 前回の平均報酬を更新
            self.before_average = ave
            # バッファを空にする
            self.buffer.clear() 

        self.total_reward = 0
        return self.env.reset(**kwargs)


    def step(self, action):
        if self.cripple_mask is not None:
            joint_mask = [i for i,x in enumerate(self.cripple_mask) if x == 99]
            
            # ver1:ランダムに選ばれた足一本の2個ある関節のうちどちらかのactuatorを変更する処理
            if joint_mask != []:
                # グラフ作成用  
                index = int(self.joint_range*10)
                if index == 10:
                    index = 9
                
                # self.joint_num：0（第一関節故障），1（第二関節故障），2（二つの関節両方故障）
                #self.joint_num = np.random.randint(0,3) # これからやろうとしている
                self.joint_num = 2
                if self.joint_num == 0:
                    action[joint_mask[0]]=henkan(action[joint_mask[0]], -1, 1, -self.joint_range, self.joint_range)
                elif self.joint_num == 1:
                    action[joint_mask[1]]=henkan(action[joint_mask[1]], -1, 1, -self.joint_range, self.joint_range)
                else:
                    action[joint_mask[0]]=henkan(action[joint_mask[0]], -1, 1, -self.joint_range, self.joint_range)
                    action[joint_mask[1]]=henkan(action[joint_mask[1]], -1, 1, -self.joint_range, self.joint_range)

        obs,reward,done,info = self.env.step(action)
        self.num_step += 1
        self.total_reward += reward

        # グラフ作成用
        joint_actuator_range_max.append(self.joint_max)
        joint_actuator_range_min.append(self.joint_min)

        self.cReward += reward
        
        return obs, reward, done, info

    def reset_task(self, value=None):
        self.crippled_leg = value if value is not None else np.random.randint(0,5)
        
        self.joint_range = random.uniform(self.joint_min, self.joint_max)

        
        # Pick which actuators to disable
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

    def visualize_fig(self, save_path):
        # figure
        figdir = "./fig/Curriculum"
        os.makedirs(figdir, exist_ok=True)
        
        plt.figure()

        fig, (ax3,ax4) = plt.subplots(1,2,figsize=(25,10))
        ax3.plot(joint_actuator_range_max)
        ax3.set(xlabel='timesteps',ylabel='k_max')
        ax4.plot(joint_actuator_range_min)
        ax4.set(xlabel='timesteps',ylabel='k_min')
        plt.savefig(figdir + "/" + "parameter_fluctuation-" + save_path)

    def output_csv(self, save_path, seed):
        # csv
        csvdir = "./output/csv/reward"
        os.makedirs(csvdir, exist_ok=True)
        R = np.array(rewardlist)
        np.savetxt(csvdir + '/'+ save_path +'-'+str(seed) +'.csv', R, delimiter=',')
        
        csvdir2 = "./output/csv/joint_actuator_range"
        os.makedirs(csvdir2, exist_ok=True)
        np.savetxt(csvdir2+'/'+ save_path +'-'+ 'joint_max-' +str(seed) +'.csv',joint_actuator_range_max,delimiter=',')
        np.savetxt(csvdir2+'/'+ save_path +'-'+ 'joint_mix-' +str(seed) +'.csv',joint_actuator_range_min,delimiter=',')