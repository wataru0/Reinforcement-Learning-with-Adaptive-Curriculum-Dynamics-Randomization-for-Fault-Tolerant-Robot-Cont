from pathlib import Path
import gym
import argparse

from rl4robot.agents import ActorCritic, PpoAgent, PpoTrainer
from rl4robot.common.evaluating_loop import EvaluatingLoop
from rl4robot.common.loggers import (
    ConsoleLogger,
    LoggerList,
    TensorBoardLogger,
)
from rl4robot.common.training_loop import TrainingLoop
from rl4robot.envs import GymEnv

import gym_custom
# from algorithms import ACDR
from algorithms import CDR, UDR, LinearCurriculumLearning
from algorithms.ACDR.ACDR import ACDREnv
from algorithms.ACDR.ACDR_training_loop import ACDRTrainingLoop
from algorithms.ACDR.ACDRB import ACDRBEnv
from algorithms.ACDR.ACDR_training_loop_by_bayes import ACDRTrainingLoopByBayes

def arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train', help='If you want to train', default=False, action='store_true')
    parser.add_argument('--agent_id', help='agent name for training', type=str, default='test')
    parser.add_argument('--eval', help='If you want to evaluation', default=False, action='store_true')
    parser.add_argument('--algo', help='train algorithm', type=str, choices=['Baseline', 'UDR', 'CDR-v1', 'CDR-v2', 'LCL-v1', 'LCL-v2', 'acdr', 'acdrb'])
    parser.add_argument('--bound_fix', help='If you want to fix lower/upper bound in train, use', default=False, action='store_true')
    parser.add_argument('--seed', help='seed for saigensei', type=int, default=1)
    
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
agent_path = "actor_critic" + "_seed={}".format(args.seed) + ".pth"
actor_critic_path = out_dir / "trained" / agent_path
# 訓練
training_steps = int(2e6)  # 1M, default: int(16e6)
tb_path = "seed={}".format(args.seed)
tensorboard_dir = out_dir / "tb" / tb_path
horizon = 2048
minibatch_size = 64
num_epochs = 10
adam_rl = 0.00022
# LCDR用
n_level = 11
# 評価
evaluating_episodes = 100
evaluating_id = args.agent_id
evaluating_dir = out_dir / "eval" / evaluating_id
video_dir = evaluating_dir / "video"
result_json = evaluating_dir / "result.json"
num_videos = 10
# ================


def backup_run_py(is_train: bool):
    import os
    import shutil

    backup_dir = out_dir if is_train else evaluating_dir
    os.makedirs(backup_dir, exist_ok=True)

    shutil.copy(__file__, backup_dir)


def save_evaluating_result_json(result):
    import json

    result_dict = {
        "num_episodes": evaluating_episodes,
        "episode_return": {
            "mean": result.episode_length_mean(),
            "std": result.episode_length_std(),
            "raw_values": [result.episode_returns],
        },
        "episode_length": {
            "mean": result.episode_length_mean(),
            "std": result.episode_length_std(),
            "raw_values": [result.episode_lengths],
        },
    }

    with open(result_json, "w") as fp:
        json.dump(result_dict, fp, indent=2)


def save_evaluating_result_tb(result):
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(tensorboard_dir)

    writer.add_hparams(
        {
            "training_id": training_id,
            "evaluating_id": evaluating_id,
        },
        {
            "eval/episode_return_mean": result.episode_return_mean(),
            "eval/episode_return_std": result.episode_return_std(),
            "eval/episode_length_mean": result.episode_length_mean(),
            "eval/episode_length_std": result.episode_length_std(),
        },
        run_name=evaluating_id,
    )


def save_evaluating_result(result):
    save_evaluating_result_json(result)
    save_evaluating_result_tb(result)


def train():
    backup_run_py(is_train=True)
    args = arg_parser()

    logger = LoggerList([ConsoleLogger(), TensorBoardLogger(tensorboard_dir)])

    # 環境を作成
    env = gym.make(env_id)
    if args.algo == "UDR":
        env = UDR.UDREnv(env)

    elif args.algo == "CDR-v1":
        env = CDR.CDREnv(env, version=1, bound_fix=args.bound_fix)
        
    elif args.algo == "CDR-v2":
        env = CDR.CDREnv(env, version=2, bound_fix=args.bound_fix)
    
    elif args.algo == "LCL-v1":
        env = LinearCurriculumLearning.LCLEnv(env, version=1, bound_fix=args.bound_fix, total_timestep=training_steps, n_level=n_level)
    
    elif args.algo == "LCL-v2":
        env = LinearCurriculumLearning.LCLEnv(env, version=2, bound_fix=args.bound_fix, total_timestep=training_steps, n_level=n_level)
    
    elif args.algo == 'acdr':
        env = ACDREnv(env)
    
    elif args.algo == 'acdrb':
        env = ACDRBEnv(env)

    env = GymEnv(env, seed=args.seed)

    # エージェントを作成
    actor_critic = ActorCritic(
        env.spec.observation_size,
        env.spec.action_size,
        actor_mlp_hidden_sizes,
        value_mlp_hidden_sizes,
    )
    actor_critic.to(device)
    trainer = PpoTrainer(actor_critic, 
        seed=args.seed, 
        horizon=horizon, 
        minibatch_size=minibatch_size,
        num_epochs=num_epochs,
        adam_lr=adam_rl)

    # 訓練
    loop = TrainingLoop(env, trainer, training_steps, logger=logger)
    if args.algo == 'acdr':
        loop = ACDRTrainingLoop(env, trainer, training_steps, logger=logger, actor_critic=actor_critic)
    elif args.algo == 'acdrb':
        loop = ACDRTrainingLoopByBayes(env, trainer, training_steps, logger=logger, actor_critic=actor_critic)
    loop.run()

    # 保存
    actor_critic.save_state(actor_critic_path)
    if args.algo == 'acdr':
        loop.save_grid_log(out_dir, seed=args.seed)
    elif args.algo == 'acdrb':
        loop.save_k_log(out_dir, seed=args.seed)

    if "CDR" in args.algo:
        CDR.CDREnv.visualize_fig(env, save_path=str(args.agent_id) + '-seed' + str(args.seed))
        CDR.CDREnv.output_csv(env, save_path=str(args.agent_id), seed=str(args.seed))


def evaluate():
    backup_run_py(is_train=False)
    args = arg_parser()

    # 環境を作成
    env = GymEnv(gym.make(env_id), seed=args.seed)

    # エージェントを作成
    actor_critic = ActorCritic(
        env.spec.observation_size,
        env.spec.action_size,
        actor_mlp_hidden_sizes,
        value_mlp_hidden_sizes,
    )
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
    print("---- result ----")
    episode_return_mean = result.episode_return_mean()
    print(
        f"average episode return (n = {evaluating_episodes}): {episode_return_mean}"
    )
    print("----------------")

    # 結果の保存
    save_evaluating_result(result)

if __name__ == "__main__":
    args = arg_parser()

    if args.train:
        train()
    if args.eval:
        evaluate()
