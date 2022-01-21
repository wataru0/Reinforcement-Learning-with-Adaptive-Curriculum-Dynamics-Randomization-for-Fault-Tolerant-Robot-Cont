from pathlib import Path
import gym
import argparse
import numpy as np
import random
from gym import wrappers
from tqdm import tqdm
import os

from rl4robot.agents import ActorCritic, PpoAgent, PpoTrainer
from rl4robot.common.evaluating_loop import EvaluatingLoop
from rl4robot.common.loggers import (
    ConsoleLogger,
    LoggerList,
    TensorBoardLogger,
)
from rl4robot.common.training_loop import TrainingLoop
from rl4robot.envs import GymEnv
from rl4robot.envs import EnvWrapper

import gym_custom
# from algorithms import ACDR
from algorithms import CDR, UDR, LinearCurriculumLearning

def arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--agent_id', help='agent name for training', type=str, default='baseline-v2')
    parser.add_argument('--type', help='select evaluation type. Barplot or generalization.', type=str, default='bar', choices=['bar', 'gene'])

    return parser.parse_args()

args = arg_parser()
# 設定
# ================
env_id = "CustomAnt-v0"
device = "cpu"
actor_mlp_hidden_sizes = [64, 64]
value_mlp_hidden_sizes = [64, 64]
training_id = args.agent_id
out_dir = Path("out") / training_id

# 評価
evaluating_episodes = 50
evaluating_id = args.agent_id
evaluating_dir = out_dir / "eval" 
video_dir = evaluating_dir / "video"
num_videos = 10
k_max = 0.5
k_min = 0.0
generalization_evaluating_episodes = 50
# ================

# generalizationを評価するための環境
class EvaluatingGeneEnv(gym.Wrapper):
    def __init__(self, env, k=1.0):
        super().__init__(env)
        self.crippled_leg = 0
        self.cripple_mask = np.ones(self.action_space.shape)
        self._init_geom_rgba = self.model.geom_rgba.copy()
        self.k = k

    def reset(self,**kwargs): #**kwargs:任意個数の引数を辞書として受け取る
        self.reset_task()
        return self.env.reset(**kwargs)

    def step(self, action):
        if self.cripple_mask is not None:
            joint_mask = [i for i,x in enumerate(self.cripple_mask) if x == 99] # 99が入っているインデックスを取得
            if len(joint_mask) == 1:
                joint_mask.append(joint_mask[0])
            # print(joint_mask) # [4,5]のように表示される, [2, 3, 4, 5]

            if joint_mask != []:
                for i in joint_mask:
                    action[i] = action[i] * (float(self.k)/100)

        obs,reward,done,info = self.env.step(action)

        return obs,reward,done,info

    def reset_task(self,value=None):
        # randomly cripple leg (4 is nothing)
        self.crippled_leg = value if value is not None else np.random.randint(0,4)

        # Pick which actuators to disable
        # joint rangeを変更する脚2本をマスクで表現、99を代入しておく
        # 壊す脚を選択
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
        

# 平均報酬，平均歩行距離の棒グラフを作成するための環境
class EvaluatingEnv(gym.Wrapper):
    def __init__(self, env, value=None, k_max=1.0, k_min=0.0):
        super().__init__(env) 
        self.value = value 
        self.crippled_leg = 0
        self.cripple_mask = np.ones(self.action_space.shape)
        self._init_geom_rgba = self.model.geom_rgba.copy()
        self.joint_range = 1.0
        self.k_max = k_max
        self.k_min = k_min

    def reset(self,**kwargs): 
        self.reset_task()
        return self.env.reset(**kwargs)

    def step(self,action):
        if self.cripple_mask is not None:
            joint_mask = [i for i,x in enumerate(self.cripple_mask) if x == 99] # 99が入っているインデックスを取得
            #print(joint_mask) # [4,5]のように表示される
            if joint_mask != []:
                action[joint_mask[0]] = action[joint_mask[0]] * self.joint_range
                action[joint_mask[1]] = action[joint_mask[1]] * self.joint_range

        obs, reward, done, info = self.env.step(action)
        
        return obs,reward,done,info

    def reset_task(self,value=None):
        # randomly cripple leg (4 is nothing)
        self.crippled_leg = value if value is not None else np.random.randint(0,4) # (0,4)だと0から4個なので0,1,2,3までしかでない！！
        
        self.joint_range = random.uniform(self.k_min, self.k_max)

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
        self.model.geom_rgba[:] = geom_rgba #[:]がないとエラーになる

def backup_run_py(is_train: bool):
    import os
    import shutil

    backup_dir = out_dir if is_train else evaluating_dir
    os.makedirs(backup_dir, exist_ok=True)

    shutil.copy(__file__, backup_dir)


def save_evaluating_result_json(result, result_json):
    import json

    result_dict = {
        "num_episodes": evaluating_episodes,
        "episode_return": {
            "mean": result.episode_return_mean(),
            "std": result.episode_return_std(),
            "raw_values": [result.episode_returns],
        },
        "episode_forward_reward": {
            "mean": result.episode_forward_reward_mean(),
            "std": result.episode_forward_reward_std(),
            "raw_values": [result.episode_forward_rewards],
        },
        "episode_length": {
            "mean": result.episode_length_mean(),
            "std": result.episode_length_std(),
            "raw_values": [result.episode_lengths],
        },
    }

    with open(result_json, "w") as fp:
        json.dump(result_dict, fp, indent=2)

def save_evaluating_result(result, result_json):
    save_evaluating_result_json(result, result_json)

def evaluate():
    backup_run_py(is_train=False)
    args = arg_parser()

    print("=====[{}]=====".format(evaluating_id))
    for seed in range(1, 6):
        # === plain env ===
        env = GymEnv(gym.make(env_id), seed=seed+100)

        # エージェントを作成
        actor_critic = ActorCritic(
            env.spec.observation_size,
            env.spec.action_size,
            actor_mlp_hidden_sizes,
            value_mlp_hidden_sizes,
        )

        agent_path = "actor_critic" + "_seed={}".format(seed) + ".pth"
        actor_critic_path = out_dir / "trained" / agent_path

        actor_critic.load_state(actor_critic_path)
        actor_critic.to(device)
        agent = PpoAgent(actor_critic)

        # 評価
        loop = EvaluatingLoop(
            env,
            agent,
            evaluating_episodes,
            video_dir=video_dir,
            num_videos=num_videos,
        )
        result = loop.run()

        # 結果の出力
        print("---- plain result ----")
        episode_return_mean = result.episode_return_mean()
        print(
            f"average episode return (n = {evaluating_episodes}): {episode_return_mean}"
        )
        print("----------------")

        # 結果の保存
        result_json = evaluating_dir / "plain_result_seed={}.json".format(seed)
        save_evaluating_result(result, result_json=result_json)

        # === broken env ===
        env = gym.make(env_id)
        env = GymEnv(EvaluatingEnv(env, k_max=k_max, k_min=k_min))
        

        # 評価
        loop = EvaluatingLoop(
            env,
            agent,
            evaluating_episodes,
            video_dir=video_dir,
            num_videos=num_videos,
        )
        result = loop.run()

        # 結果の出力
        print("---- broken result ----")
        episode_return_mean = result.episode_return_mean()
        print(
            f"average episode return (n = {evaluating_episodes}): {episode_return_mean}"
        )
        print("----------------")

        # 結果の保存
        result_json = evaluating_dir / "broken_result_seed={}.json".format(seed)
        save_evaluating_result(result, result_json=result_json)

def evaluate_generarization():
    args = arg_parser()
    nd_dir = './out/' + str(training_id) + "/gene/"
    os.makedirs(nd_dir, exist_ok=True)

    print("=====[{}]=====".format(evaluating_id))
    for seed in range(1, 6):
        # seed毎のkに対する平均エピソード収益を格納する(k)の配列
        k_seed_gene = np.zeros(100)

        for k in tqdm(range(0,100)):
            env = gym.make(env_id)
            env = GymEnv(EvaluatingGeneEnv(env, k), seed=seed+100)

            # エージェントを作成
            actor_critic = ActorCritic(
                env.spec.observation_size,
                env.spec.action_size,
                actor_mlp_hidden_sizes,
                value_mlp_hidden_sizes,
            )

            agent_path = "actor_critic" + "_seed={}".format(seed) + ".pth"
            actor_critic_path = out_dir / "trained" / agent_path

            actor_critic.load_state(actor_critic_path)
            actor_critic.to(device)
            agent = PpoAgent(actor_critic)
            
            # 評価
            loop = EvaluatingLoop(
                env,
                agent,
                generalization_evaluating_episodes,
                gene_flag=True
            )
            result = loop.run()

            k_seed_gene[k] = result.episode_return_mean()
        
        np.save(nd_dir + "seed={}".format(seed), k_seed_gene)

if __name__ == "__main__":
    args = arg_parser()
    if args.type == 'bar':
        evaluate()
    elif args.type == 'gene':
        evaluate_generarization()
