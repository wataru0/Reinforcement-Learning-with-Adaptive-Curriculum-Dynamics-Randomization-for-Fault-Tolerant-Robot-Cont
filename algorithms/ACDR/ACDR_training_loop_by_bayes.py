# 2021/12/18
# ACDRアルゴリズムにベイズ最適化を組み合わせた学習を行う訓練ループ処理

import time
from typing import Final, List, Optional, Tuple, Union
import numpy as np
import torch
import gym
import sys
import json
import GPyOpt

sys.path.append('.../')
from rl4robot.agents.ppo import actor_critic, trainer
from rl4robot.agents import Trainer, ActorCritic, PpoAgent
from rl4robot.envs import Env
from rl4robot.types import ObservationArray
from rl4robot.common.training_loop import TrainingLoop
from rl4robot.common.evaluating_loop import EvaluatingLoop
from rl4robot.common.loggers import Logger

from algorithms.ACDR import ACDR
from algorithms.Baseline import BaselineEnv
import gym_custom

__all__ = [
    "ACDRTrainingLoopByBayes",
]

# hyperparameters
# ======
evaluating_episodes = 10
# ======

# # 真ACDRプロトタイプ　訓練ループ
class ACDRTrainingLoopByBayes(TrainingLoop):
    def __init__(self, env: Env, trainer: Trainer, num_steps: int, actor_critic: ActorCritic, maximize_flag = True, logger: Optional[Logger] = None) -> None:
        super().__init__(env, trainer, num_steps, logger=logger)

        # --インスタンス変数--
        self.actor_critic = actor_critic
        # ベイズ最適化で求めたkの値を記録する変数
        self.k_opt = 0.0

        # 学習中にベイズ最適化により選択されたkを記録するdict
        self.k_log = {}

        # ベイズ最適化で最大化を行うか，最小化を行うかのフラグ
        self.maximize_flag = maximize_flag

    # _update()はPPOのモデル更新時に呼ばれる
    def _update(self):
        super()._update()
        # 各グリッドでの性能を推定する
        # self._estimate_capability()

        # ベイズ最適化によるサンプリング確率の更新
        self._update_k_sampling_value_by_bayes()

        progress = self.global_step / self.num_steps
        # progress = round(progress, 2)
        # progress = math.floor(progress * 10**2) / (10**2)
        # if int(progress % 0.01) == 0:
        #     self.k_log[progress] = self.k_opt.copy()
        #     # print(self.k_log)
        self.k_log[progress] = self.k_opt.copy()

    def save_k_log(self, path, seed=1):
        """
        k_sampling_gridをjson形式で保存
        """

        print(self.k_log)
        filename = 'k_log_seed-' + str(seed) + '.json'
        with open(path / filename, 'w') as f:
            json.dump(self.k_log, f, indent=4)

    def _record_log(self):
        super()._record_log()
        

    # ベイズ最適化によるサンプリング確率の更新
    def _update_k_sampling_value_by_bayes(self):
        def run_estimate_capability(*args):
            agent = PpoAgent(self.actor_critic)
            action_low, action_hight = self.env.spec.action_range

            env = gym.make('CustomAnt-v0')
            est_env = BaselineEnv(env, k=args[0])
            observation = est_env.reset()

            episode_done = False
            episode_length = 0
            episode_return = 0.0
            capab = 0
            while True:
                if episode_done:
                    est_env.reset()
                    break
                action = agent.act(observation)
                action = np.clip(action, action_low, action_hight)
                observation, reward, episode_done, info = est_env.step(action)

                # ndarray -> tensor
                obs = torch.from_numpy(observation)
                capab += self.actor_critic.get_values(obs).detach().cpu().numpy().copy()
                
                episode_length += 1
                episode_return += reward

            return capab

        bounds = [{'name': 'k', 
                    'type': 'continuous',
                    'domain': (0.0, 1.5)}]

        # Run bayse optimization
        # maximize: Trueの時最大化，Falseの時最小化
        my_opt = GPyOpt.methods.BayesianOptimization(f=run_estimate_capability, domain=bounds, maximize=self.maximize_flag)
        my_opt.run_optimization(max_iter=10)

        # ベイズ最適化により求まったkの値（self.k_opt）
        self.k_opt = my_opt.x_opt[0]
        print('k_opt: ', self.k_opt)
        # 環境にk_optを適用する
        self.env.set_k(self.k_opt)
        