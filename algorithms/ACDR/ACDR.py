# Adaptive Curriculum Dynamics Randomization Algorithm
# 2021/10/16
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

class ACDREnv(gym.Wrapper):
# class ACDREnv(EnvWrapper, gym.wrappers.TimeLimit):
# class ACDREnv(EnvWrapper):
# class ACDREnv(gym.wrappers.TimeLimit):
    def __init__(self, env, num_grids=10, k_low=0.0, k_hight=1.5, value=None):
        # env = GymEnv(env)
        super().__init__(env)

        # 以下インスタンス変数
        self.value = value
        self.crippled_leg = 0
        self.cripple_mask = np.ones(self.action_space.shape)
        # self.cripple_mask = np.ones(self.gym_env.action_space.shape)
        # self._init_geom_rgba = self.model.geom_rgba.copy()
        self.joint_range = 1.0
        self.num_step = 0
        self.total_reward = 0
        # 累積報酬
        self.cReward = 0 
        # 各関節の故障率を格納するリスト,action_spaceと同じ大きさで1に初期化
        self.actuator_list = np.ones(self.action_space.shape)
        # self.actuator_list = np.ones(self.gym_env.action_space.shape)


        # kのサンプリング範囲を選択するときの重み付きリスト
        # 徐々に得意なrangeの値をインクリメントしていく
        # 学習初期[0,0,0,0,0] -> [1,0,4,5,6] -> 学習終盤[10,8,9,10]
        # self.k_range_sampling_grid = k_range_sampling_grid
        # num_grids分の配列を作る
        self.num_grids = num_grids
        self.k_sampling_grid = [1 for _ in range(num_grids)]

        # ランダム化パラメータkの最小値と最大値の設定
        self.k_low = k_low
        self.k_hight = k_hight

        # dr分布
        self.k_range = (k_low, k_hight)

        # capabilityをestimateしているかどうかのフラグ
        self.not_estimating_flag = True

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
                action[joint_mask[0]]=henkan(action[joint_mask[0]], -1, 1, -self.joint_range, self.joint_range)
                action[joint_mask[1]]=henkan(action[joint_mask[1]], -1, 1, -self.joint_range, self.joint_range)
    
        obs,reward,done,info = self.env.step(action)
        self.num_step += 1
        self.total_reward += reward

        # グラフ作成用
        # joint_actuator_range_max.append(self.joint_max)
        # joint_actuator_range_min.append(self.joint_min)

        self.cReward += reward
        
        return obs, reward, done, info

    # 新しいサンプリンググリッドをセットするメソッド
    # 訓練ループ（このファイルの外）から利用される
    def set_k_sampling_grid(self, list):
        self.k_sampling_grid = list

    # 適用するk_rangeを直接指定するメソッド
    # こいつを使ってestimate_capabilityする
    def set_grid(self, grid):
        """
        k_rangeは要素2のタプル，(k_min, k_max)
        """
        range_step = (self.k_hight - self.k_low)/self.num_grids
        k_min = grid*range_step
        k_max = (grid+1)*range_step

        # self.k_range = (k_min, k_max)
        self.joint_range = random.uniform(k_min, k_max)

    def set_not_estimating_flag(self, flag):
        self.not_estimating_flag = flag

    # kをサンプリングする範囲を決定するメソッド
    # 累積和を利用した重み付きサンプリングを行う https://colab.research.google.com/drive/1Ls3RYWB7-0XwVAri2GHPwBIg8H4oCBQJ?usp=sharing#scrollTo=g-jewqjnuCDt
    def _select_sampling_range(self):
        # self.k_range_sampling_gridには重みが格納されている：[1,2,3,4,5]や[10,10,10,90]のように，
        # その重みに応じてサンプリングを行う範囲を決定する
        # weight_sum = sum(self.k_sampling_grid)
        # r = random.random() * weight_sum
        
        # # サンプリングするindexを取得
        # num = 0
        # for i, weight in enumerate(self.k_sampling_grid):
        #     num += weight
        #     if r <= num:
        #         sampling_index = i
        #         break

        sampling_index_list = [i for i in range(self.num_grids)]
        sampling_index = random.choices(sampling_index_list, weights=self.k_sampling_grid)[0]
        
        # indexからサンプリングするk_rangeを求める
        range_step = (self.k_hight - self.k_low)/self.num_grids
        k_min = sampling_index*range_step
        k_max = (sampling_index+1)*range_step
        
        self.k_range = (k_min, k_max)

    def reset_task(self, value=None):
        # kの範囲(k_range)を重み付き確率分布から選択
        self._select_sampling_range()

        # randomly cripple leg (4 is nothing)
        # (0,4)だと0から4個なので0,1,2,3までしかでない！！
        self.crippled_leg = value if value is not None else np.random.randint(0,5)
        
        # joint_min~joint_maxまでの乱数を生成。これがaction値のrangeになる
        # self.joint_range = random.uniform(self.joint_min, self.joint_max)
        k_min, k_max = self.k_range
        self.joint_range = random.uniform(k_min, k_max)
        
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