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

def arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train', help='If you want to train', default=False, action='store_true')
    parser.add_argument('--training_id', help='agent name for training', type=str, default='test')
    parser.add_argument('--eval', help='If you want to evaluation', default=False, action='store_true')
    parser.add_argument('--evaluating_id', help='agent name for evalation', type=str, default='test')
    parser.add_argument('--seed', help='seed for saigensei', type=int, default=1)
    
    return parser.parse_args()

args = arg_parser()
# 設定
# ================
env_id = "CustomAnt-v0"
device = "cpu"
actor_mlp_hidden_sizes = [64, 64]
value_mlp_hidden_sizes = [64, 64]
training_id = args.training_id
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
# 評価
evaluating_episodes = 100
evaluating_id = args.evaluating_id
evaluating_dir = out_dir / "eval" / evaluating_id
video_dir = evaluating_dir / "video"
result_json = evaluating_dir / "result.json"
num_videos = 10
# ================


# # カリキュラム学習付きの訓練ループ
# class CurriculumTrainingLoop(TrainingLoop):
#     # _update()はPPOのモデル更新時（=horizonステップごと）に呼ばれる
#     def _update(self):
#         assert isinstance(self.env, McmcTerrainEnv)

#         super()._update()

#         # ==== MCMC地形のカリキュラム学習 ====
#         # 注意！！
#         # この実装はスコアに応じて適用的に j を変化させるものではなく、
#         # 時刻に応じて j を変化させるもの

#         progress = self.global_step / self.num_steps  # 訓練の進捗（0.0 〜 1.0）

#         # TODO: 進捗に応じて j を決定する
#         j = 0.5

#         # j を変更
#         # 次のエピソードからこの j が適用される
#         self.env.set_j(j)

#         # ================

#     def _record_log(self):
#         super()._record_log()

#         self._record_curriculum_log()

#     def _record_curriculum_log(self):
#         if self.logger:
#             assert isinstance(self.env, McmcTerrainEnv)

#             self.logger.record("curriculum/j", self.env.j)


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
    env = GymEnv(gym.make(env_id), seed=args.seed)

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
    loop.run()

    # 保存
    actor_critic.save_state(actor_critic_path)


def evaluate():
    backup_run_py(is_train=False)
    args = arg_parser()

    # 環境を作成
    env = GymEnv(gym.make(env_id),seed=args.seed)

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
