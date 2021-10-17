import os
from pathlib import Path

import gym

from rl4robot.agents import ActorCritic, PpoAgent, PpoTrainer
from rl4robot.common.evaluating_loop import EvaluatingLoop
from rl4robot.common.loggers import (
    ConsoleLogger,
    LoggerList,
    TensorBoardLogger,
)
from rl4robot.common.training_loop import TrainingLoop
from rl4robot.envs import GymEnv

# 設定
# ================
env_id = "Walker2d-v2"
device = "cpu"
actor_mlp_hidden_sizes = [64, 64]
value_mlp_hidden_sizes = [64, 64]
out_dir = Path("out")
actor_critic_path = out_dir / "trained" / "actor_critic.pth"
# 訓練
training_steps = 2 ** 20  # 1M
tensorboard_dir = out_dir / "tb"
# 評価
evaluating_episodes = 100
video_dir = out_dir / "video"
num_videos = 10
# ================


def train() -> None:
    logger = LoggerList([ConsoleLogger(), TensorBoardLogger(tensorboard_dir)])

    # 環境を作成
    env = GymEnv(gym.make(env_id))

    # エージェントを作成
    actor_critic = ActorCritic(
        env.spec.observation_size,
        env.spec.action_size,
        actor_mlp_hidden_sizes,
        value_mlp_hidden_sizes,
    )
    actor_critic.to(device)
    trainer = PpoTrainer(actor_critic)

    # 訓練
    loop = TrainingLoop(env, trainer, training_steps, logger=logger)
    loop.run()

    # 保存
    actor_critic.save_state(actor_critic_path)


def evaluate() -> None:
    # 環境を作成
    env = GymEnv(gym.make(env_id))

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


def main() -> None:
    os.makedirs(out_dir)
    train()
    evaluate()


if __name__ == "__main__":
    main()
