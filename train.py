# 2021/04/27
# train用プログラム
# リファクタリングした

# Execute command -------------
# train.py --savedir=(保存するモデルの名前) --seed=(乱数のseed値)
# -----------------------------

import os
import gym
import numpy as np
import matplotlib.pyplot as plt
import argparse
import datetime
import random
import time
import pytz
import tensorflow as tf
import seaborn as sns; sns.set()
import pandas as pd

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results,ts2xy
from stable_baselines import results_plotter
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2, SAC
from stable_baselines.common.vec_env import DummyVecEnv

import gym_custom
from algorithms import CDR, UDR, LinearCurriculumLearning, TriangularDR, Baseline

best_mean_reward = -np.inf # 平均報酬
n_updates = 0 # 更新数
log_dir = "./monitor_log/"
os.makedirs(log_dir,exist_ok=True)

# ---------------------parameter discription --------------------------------
# total_timesteps is the number of steps in total the agent will do for any 
# environment. The total_timesteps can be across several episodes,
# n_updates = total_timesteps // self.n_batch 
# n_batchは、n_stepsにベクトル化された環境の数を掛けたものです。
# つまり、n_stepを32に設定し、total_timesteps = 25000で1つの環境を実行する場合、ラーニングコール中にポリシーを781更新します（PPOは1つのバッチで複数の更新を実行できるため）
# https://stackoverflow.com/questions/56700948/understanding-the-total-timesteps-parameter-in-stable-baselines-models
# https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
# ---------------------------------------------------------------------------

# env以外はppo用のパラメータ
config = {
    'env':'CustomAnt-v0',
    # 'env':'Ant-v2',
    # 'env':'AblationAnt-v0', # for ablation study
    'total_timestep':int(16e6), # default:16e6, PPO-PytorchのN_updatesとは違い、単純に訓練に使われる総タイムステップ数 
    'n_steps':128, # ポリシー更新前に収集する経験の数(ステップ数)
    'nminibatches':4, # 勾配降下に使うミニバッチのサイズ
    'noptepochs':4, # 収集した経験を勾配降下にかける回数
    'learning_rate':0.00022, # 0.00020
    'n_level':11, # Linear Curriculum Learningにおいてのkの更新回数
}

def arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--savedir', help='saving name dir for trained mode!!!', type=str, default='Ant'+ datetime.datetime.now().isoformat())
    parser.add_argument('--seed', help='seed for saigensei', type=int, default=1)
    parser.add_argument('--algo', help='train algorithm', type=str, choices=['Baseline', 'UDR', 'CDR-v1', 'CDR-v2', 'LCL-v1', 'LCL-v2', 'TDR'], required=True)
    parser.add_argument('--ablation', help='Do you want to do ablation study? hahaha.', default=False, action='store_true')
    parser.add_argument('--bound_fix', help='If you want to fix lower/upper bound in train, use', default=False, action='store_true')
    parser.add_argument('--baseline_k', help='If you want to fix k of baseline method, set!', type=float, default=1.0)
    parser.add_argument('--RL_algo', help='Chose RL algorithms', type=str, default='ppo', choices=['ppo', 'sac'])

    return parser.parse_args()

def callback(_locals,_globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global n_updates
    global best_mean_reward
    # Print stats every 1000 calls
    if (n_updates + 1) % 1000 == 0:
        # Evaluate policy training performance
        x,y = ts2xy(load_results(log_dir),'timesteps')
        if len(x) > 0:
            #100ステップ毎に過去100件の平均報酬を計算し、ベスト平均報酬を越えていたらエージェントを保存しています。
            mean_reward = np.mean(y[-100:])

            # New best model, you could save the agent here
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # Example for saving best model
                print("Saving new best model")
                _locals['self'].save(log_dir + 'best_model.pkl')
   
    n_updates += 1
    return True

def main():
    args = arg_parser()

    # Set trained agent dir and tensorbord dir
    if args.ablation is not True:
        save_dir = "./trained_agent_dir/"+ args.savedir + "/"
        # Create dir
        os.makedirs(save_dir,exist_ok=True)

    # HDDに保存する
    home = str(os.environ['HOME'])
    tensorboard_log_dir = home + "/HDD/RA-L/tensorboard_log3/"
    
    if args.ablation:
        # tensorboard_log_dir = "./Ablation/tensorboard_log/"
        tensorboard_log_dir = home + "/HDD/RA-L/Ablation_tensorboard_log2/"
        
    # Create dir
    os.makedirs(tensorboard_log_dir,exist_ok=True)

    # Create gym environment
    env = gym.make(config['env'])
    if args.algo == "UDR":
        print("Now, we are training the agent using the UDR method!")
        env = UDR.UDREnv(env)

    elif args.algo == "CDR-v1":
        print("Now, we are training the agent using the CDR-v1 method!")
        env = CDR.CDREnv(env, version=1, bound_fix=args.bound_fix)
        
    elif args.algo == "CDR-v2":
        print("Now, we are training the agent using the CDR-v2 method!")
        env = CDR.CDREnv(env, version=2, bound_fix=args.bound_fix)
    
    elif args.algo == "LCL-v1":
        env = LinearCurriculumLearning.LCLEnv(env, version=1, bound_fix=args.bound_fix, total_timestep=config['total_timestep'], n_level=config['n_level'])
    
    elif args.algo == "LCL-v2":
        env = LinearCurriculumLearning.LCLEnv(env, version=2, bound_fix=args.bound_fix, total_timestep=config['total_timestep'], n_level=config['n_level'])

    elif args.algo == "TDR":
        env = TriangularDR.TDREnv(env)

    else:
        print("Now, we are training the agent using the baseline method!")
        env = Baseline.BaselineEnv(env, k=args.baseline_k)

    env = Monitor(env, log_dir, allow_early_resets=True) 
    env = DummyVecEnv([lambda :env])

    # create model and train
    if args.RL_algo == 'ppo':
        model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=tensorboard_log_dir, n_steps=config['n_steps'], nminibatches=config['nminibatches'], noptepochs=config['noptepochs'], learning_rate=config['learning_rate'], seed=args.seed)
        # train model
        model.learn(total_timesteps=config['total_timestep'], callback=callback, tb_log_name=args.savedir)

    elif args.RL_algo == 'sac':
        # model = SAC(MlpPolicy, env, verbose=1, tensorboard_log=tensorboard_log_dir, n_steps=config['n_steps'], learning_rate=config['learning_rate'], seed=args.seed)
        model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log_dir)
        # train model
        model.learn(total_timesteps=int(4e6), tb_log_name=args.savedir, log_interval=10)

    
    # Save the model(agent)
    if args.ablation is not True:
        model.save(save_dir + "trainedAnt" + "-seed"+ str(args.seed))

    # CDR用の可視化処理
    if "CDR" in args.algo:
        CDR.CDREnv.visualize_fig(env, save_path=str(args.savedir) + '-seed' + str(args.seed))
        CDR.CDREnv.output_csv(env, save_path=str(args.savedir), seed=args.seed)

    if "LCL" in args.algo:
        LinearCurriculumLearning.LCLEnv.visualize_fig(env, save_path=str(args.savedir) + '-seed' + str(args.seed))
        LinearCurriculumLearning.LCLEnv.output_csv(env, save_path=str(args.savedir), seed=args.seed)

if __name__ == '__main__':
    main()
        