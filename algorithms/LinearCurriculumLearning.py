# Linear Curriculum Learning Algorithm
# 2021/06/17
# 提案手法
# 今まで一つのファイルでトレーニングを回していたのをクラスごとにしっかり分けようというもの
# train.pyを実行する際に，gym環境をラッパーすることで実装する
# ver 1 は高いトルク制限値kから始めるもの
# ver 2 は低いトルク制限値kから始めるもの

# トルクのランダム化範囲について先行研究
# https://arxiv.org/pdf/2010.04304.pdf
# k は [0.0, 1.5]でやる

import os
import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd
import datetime

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

class LCLEnv(gym.Wrapper):
    def __init__(self, env, value=None, version=2, bound_fix=False, total_timestep=int(16e6), n_level=10):
        super().__init__(env)
        # 以下インスタンス変数
        self.value = value 
        self.version = version
        self.crippled_leg = 0
        self.cripple_mask = np.ones(self.action_space.shape)
        self._init_geom_rgba = self.model.geom_rgba.copy()
        self.joint_range = 1.0
        self.num_step = 0
        self.total_reward = 0
        
        self.joint_num = 0
        # 累積報酬
        self.cReward = 0 
        # 各関節の故障率を格納するリスト,action_spaceと同じ大きさで1に初期化
        self.actuator_list = np.ones(self.action_space.shape)

        # minもmaxも1からスタート(v1),minもmaxも0からスタート(v2)
        if version == 1:
            self.joint_min = 1.5
            self.joint_max = 1.5 
        elif version == 2:
            self.joint_min = 0.0 
            self.joint_max = 0.0

        # Trueの時，v1の時は上限固定，v2の時は下限固定する
        self.bound_fix = bound_fix

        # 線形にkを変化させるためにトータルタイムステップを使用
        self.total_timestep = total_timestep

        # n_level:kの範囲を何段階に分けるかを表すパラメータ，この数だけカリキュラム学習を行う
        self.n_level = n_level

        # 各パラメータの計算
        # 1 n_levelあたりに更新するkの値
        self.k_update_size = 1.5/(n_level-1)
        # kの更新タイミング
        self.k_update_point = total_timestep//n_level

    def reset(self, **kwargs): 
        self.reset_task()
        rewardlist.append(self.cReward)
        self.cReward = 0 
        self.total_reward = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        if self.cripple_mask is not None:
            joint_mask = [i for i,x in enumerate(self.cripple_mask) if x == 99] # 99が入っているインデックスを取得
            # print(joint_mask) # [4,5]のように表示される
            
            # ver1:ランダムに選ばれた足一本の2個ある関節のうちどちらかのactuatorを変更する処理
            if joint_mask != []:
                # グラフ作成用
                # 各関節の故障率の分布を見る，actuator_mapで      
                # index = int(self.joint_range*10)
                # if index == 10:
                #     index = 9
                
                # self.joint_num：0（第一関節故障），1（第二関節故障），2（二つの関節両方故障）
                #self.joint_num = np.random.randint(0,3) # これからやろうとしている
                self.joint_num = 2 # 前のやり方
                if self.joint_num == 0:
                    action[joint_mask[0]]=henkan(action[joint_mask[0]], -1, 1, -self.joint_range, self.joint_range)
                    # actuator_map[index][joint_mask[0]] += 1
                elif self.joint_num == 1:
                    action[joint_mask[1]]=henkan(action[joint_mask[1]], -1, 1, -self.joint_range, self.joint_range)
                    # actuator_map[index][joint_mask[1]] += 1
                else:
                    action[joint_mask[0]]=henkan(action[joint_mask[0]], -1, 1, -self.joint_range, self.joint_range)
                    action[joint_mask[1]]=henkan(action[joint_mask[1]], -1, 1, -self.joint_range, self.joint_range)
                    # actuator_map[index][joint_mask[0]] += 1
                    # actuator_map[index][joint_mask[1]] += 1
            #ーーーー action = self.cripple_mask * action
            #print(action) # joint_maskの要素のaction値をクリップ(指定した値の間の値に変換)することができた

        # ver2:エージェントにある関節8個全てのactuatorをランダムに変更する処理
        # action = action * self.actuator_list #np.arrayはこの計算できる，各要素同士の積になる

        obs,reward,done,info = self.env.step(action)
        self.num_step += 1
        self.update_check(num_step=self.num_step)
        self.total_reward += reward

        # グラフ作成用
        joint_actuator_range_max.append(self.joint_max)
        joint_actuator_range_min.append(self.joint_min)

        self.cReward += reward
        
        return obs, reward, done, info

    def update_check(self, num_step):
        # 更新タイミングなら更新
        if num_step % self.k_update_point == 0:
            # Curriculum-v1
            if self.version == 1:
                self.joint_min -= self.k_update_size
                # self.joint_minがマイナスになったら０にする
                self.joint_min = max(0.0, self.joint_min)
                # kのminとmaxの差が0.1以上になったらmaxも減らす (upperfixではコメントアウト)
                if self.bound_fix is not True:
                    if abs(self.joint_max - self.joint_min) > self.k_update_size:
                        self.joint_max -= self.k_update_size
                        self.joint_max = max(0.0, self.joint_max)
            # Curriculum-v2
            if self.version == 2:
                self.joint_max += self.k_update_size
                # self.joint_maxが上限値を超えたら上限値にする
                self.joint_max = min(1.5, self.joint_max)
                # kのminとmaxの差が0.1以上になったらminも上昇 (lowerfixではコメントアウト)
                if self.bound_fix is not True:
                    if abs(self.joint_max - self.joint_min) > self.k_update_size:
                        self.joint_min += self.k_update_size
                        self.joint_min = min(1.5, self.joint_min)

    def reset_task(self, value=None):
        # randomly cripple leg (4 is nothing)
        # (0,4)だと0から4個なので0,1,2,3までしかでない！！
        self.crippled_leg = value if value is not None else np.random.randint(0,5)
        
        # joint_min~joint_maxまでの乱数を生成。これがaction値のrangeになる
        self.joint_range = random.uniform(self.joint_min, self.joint_max)

        # もう少し考える必要ある---6/28
        # 各関節で変化させる
        # for i in range(len(self.actuator_list)):
        #     #self.joint_range = random.uniform(0,self.joint_max) #確認用
        #     self.joint_range = random.uniform(self.joint_min,self.joint_max)
        #     self.actuator_list[i] = self.joint_range # 各関節リストにactuatorの値を格納
        #---------------------------
        
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