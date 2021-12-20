# Adaptive Curriculum Dynamics Randomization with Bayes Algorithm
# 2021/12/18
# 提案手法
# 今まで一つのファイルでトレーニングを回していたのをクラスごとにしっかり分けようというもの
# train.pyを実行する際に，gym環境をラッパーすることで実装する

# ALLSTEPSを参考に，報酬の低い故障は発生確率を低くすることでカリキュラム学習を行う．

# トルクのランダム化範囲について先行研究
# https://arxiv.org/pdf/2010.04304.pdf

import os
import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import sys
sys.path.append('.../')
from seaborn import widgets

from rl4robot.envs import EnvWrapper, EnvSpec, EnvStep

sns.set()

#可視化用
joint_actuator_range_max = [] # kの上限値を格納するリスト
joint_actuator_range_min = [] # kの下限値を格納するリスト
actuator_map = np.zeros((10,8)) # 適用された故障率の分布を調べるためのマップ
actuator_bunpu= [0]*10
actuator_power_map = np.zeros((10,8)) # 実際のactuator出力の分布を調べるためのマップ
rewardlist = []

# 平均報酬
best_mean_reward = -np.inf 
# 更新数
n_updates = 0 

def henkan(val, start1, stop1, start2, stop2):
    return start2 + (stop2 - start2) * ((val-start1)/(stop1 - start1))

class ACDRBEnv(gym.Wrapper):
    def __init__(self, env, k_low=0.0, k_hight=1.5, value=None):
        # env = GymEnv(env)
        super().__init__(env)

        # 以下インスタンス変数
        self.value = value
        self.crippled_leg = 0
        self.cripple_mask = np.ones(self.action_space.shape)
        self.joint_range = 1.0
        self.total_reward = 0
        # 累積報酬
        self.cReward = 0 
        # 各関節の故障率を格納するリスト,action_spaceと同じ大きさで1に初期化
        self.actuator_list = np.ones(self.action_space.shape)
        # self.actuator_list = np.ones(self.gym_env.action_space.shape)

        # ランダム化パラメータkの最小値と最大値の設定
        self.k_low = k_low
        self.k_hight = k_hight

        # dr分布
        self.k_range = (k_low, k_hight)

        # capabilityをestimateしているかどうかのフラグ
        self.not_estimating_flag = True

        self.k = 1.0

    def reset(self, **kwargs): 
        if self.not_estimating_flag:
            self.reset_task()

        rewardlist.append(self.cReward)
        self.cReward = 0 
        self.total_reward = 0

        return self.env.reset(**kwargs)


    def step(self, action):
        if self.cripple_mask is not None:
            joint_mask = [i for i,x in enumerate(self.cripple_mask) if x == 99] # 99が入っているインデックスを取得
            # print(joint_mask) # [4,5]のように表示される
            
            # ランダムに選ばれた足一本の2個ある関節のactuatorを変更する処理
            if joint_mask != []:
                action[joint_mask[0]] *= self.k
                action[joint_mask[1]] *= self.k
    
        obs,reward,done,info = self.env.step(action)
        self.total_reward += reward

        self.cReward += reward
        
        return obs, reward, done, info

    # 新しいkをセットするメソッド
    # 訓練ループ（このファイルの外）から利用される
    def set_k(self, k):
        self.k = k

    def set_not_estimating_flag(self, flag):
        self.not_estimating_flag = flag

    def reset_task(self, value=None):

        # randomly cripple leg (4 is nothing)
        # (0,4)だと0から4個なので0,1,2,3までしかでない！！
        self.crippled_leg = value if value is not None else np.random.randint(0,5)
        
        # # 適用するkの値を設定
        # self.joint_range = self.k
        
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