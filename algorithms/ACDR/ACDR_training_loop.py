# 2021/11/24
# ACDRアルゴリズムにより学習を行う訓練ループ処理

import time
from typing import Final, List, Optional, Tuple, Union
import numpy as np
import torch
import gym
import sys
import json
import math

sys.path.append('.../')
from rl4robot.agents.ppo import actor_critic, trainer
from rl4robot.agents import Trainer, ActorCritic, PpoAgent
from rl4robot.envs import Env
from rl4robot.types import ObservationArray
from rl4robot.common.training_loop import TrainingLoop
from rl4robot.common.evaluating_loop import EvaluatingLoop
from rl4robot.common.loggers import Logger

from algorithms.ACDR import ACDR
import gym_custom

__all__ = [
    "ACDRTrainingLoop",
]

# hyperparameters
# ======
evaluating_episodes = 10
# ======

# # 真ACDRプロトタイプ　訓練ループ
class ACDRTrainingLoop(TrainingLoop):
    def __init__(self, env: Env, trainer: Trainer, num_steps: int, actor_critic: ActorCritic, num_grids: int = 10, logger: Optional[Logger] = None) -> None:
        super().__init__(env, trainer, num_steps, logger=logger)

        # --インスタンス変数--
        self.actor_critic = actor_critic
        # n gridに分割したランダム化パラメータのサンプリング分布のサンプリング確率を格納するリスト
        self.k_sampling_grid = [1 for _ in range(num_grids)]
        self.num_grids = num_grids

        self.grid_log = {}

    # _update()はPPOのモデル更新時に呼ばれる
    def _update(self):
        super()._update()
        # 各グリッドでの性能を推定する
        self._estimate_capability()

        progress = self.global_step / self.num_steps
        # progress = round(progress, 1)
        progress = math.floor(progress * 10) / (10)
        if int(progress % 0.1) == 0:
            self.grid_log[progress] = self.k_sampling_grid.copy()
            print(self.grid_log)

    def save_grid_log(self, path, seed=1):
        """
        k_sampling_gridをjson形式で保存
        """

        print(self.grid_log)
        filename = 'k_sampling_grid_seed-' + str(seed) + '.json'
        with open(path / filename, 'w') as f:
            json.dump(self.grid_log, f, indent=4)

    def _record_log(self):
        super()._record_log()

        # self._record_sampling_probability()

    def _record_sampling_probability(self):
        if self.logger:
            self.logger.record("sampling_probability", self.k_sampling_grid)

    # エージェントの各グリッドでの能力を推定するメソッド
    # 各グリッドでの価値関数の値を収集
    def _estimate_capability(self):
        
        agent = PpoAgent(self.actor_critic)
        action_low, action_hight = self.env.spec.action_range

        capability_of_each_grid = np.zeros(self.num_grids)

        # 各グリッドでcapabilityを求める
        for grid in range(self.num_grids):
            env = gym.make('CustomAnt-v0')
            est_env = ACDR.ACDREnv(env, num_grids=10)
            # set estimating flag 
            est_env.set_not_estimating_flag(False)

            episode_done = False
            episode_length = 0
            episode_return = 0.0

            observation = est_env.reset()
            est_env.set_grid(grid)
            # print(grid, est_env.joint_range)

            capab = 0
            # 各grid envでの評価ループ
            while True:
                if episode_done:
                    est_env.reset()
                    break
                action = agent.act(observation)
                action = np.clip(action, action_low, action_hight)
                # env_step = self.env.step(action)
                # observation = env_step.observation
                observation, reward, episode_done, info = est_env.step(action)

                # ndarray -> tensor
                obs = torch.from_numpy(observation)
                capab += self.actor_critic.get_values(obs).detach().cpu().numpy().copy()
                
                
                # episode_done = env_step.episode_done
                episode_length += 1
                episode_return += reward
                
            
            capability_of_each_grid[grid] = capab
            # print(grid, capab)
            # print(capability_of_each_grid)
            # print(episode_length, episode_return)
        
        print("capability: ", capability_of_each_grid)
        est_env.set_not_estimating_flag(True)
        self._update_k_sampling_grid(capability_of_each_grid)

    # 学習初期の+1と学習終盤の+1は分母が大きくなっているため重みが違う
    # 12/15：累積和ではなくsoftmax関数を用いる
    def _update_k_sampling_grid(self, capability_of_each_grid):
        # ===累積和による更新===
        # # 最もcapabilityが低いグリッドの発生確率を上げる
        # worst_grid = np.argmin(capability_of_each_grid)
        # self.k_sampling_grid[worst_grid] += 1

        def min_max_norm(x, axis=None):
            min = x.min(axis=axis, keepdims=True)
            max = x.max(axis=axis, keepdims=True)
            return (x - min)/(max - min)

        # ===ソフトマックス関数による更新===
        # capabilityのminmax正規化
        capability_of_each_grid = min_max_norm(capability_of_each_grid)
        print("capability_norm: ", capability_of_each_grid)
        # self.k_sampling_grid = torch.nn.functional.softmax(torch.tensor(capability_of_each_grid)/5).tolist()
        self.k_sampling_grid = torch.nn.functional.softmax(torch.tensor(capability_of_each_grid)).tolist()

        self.env.set_k_sampling_grid(self.k_sampling_grid)
