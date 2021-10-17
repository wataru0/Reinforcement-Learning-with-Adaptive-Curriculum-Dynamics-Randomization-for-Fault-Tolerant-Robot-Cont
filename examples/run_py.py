# 大里さんが作ったサンプルプログラム
# 自作環境（地形カリキュラムなど）を使った訓練を実装している

from pathlib import Path

from rl4robot.agents import ActorCritic, PpoAgent, PpoTrainer
from rl4robot.common.evaluating_loop import EvaluatingLoop
from rl4robot.common.loggers import (
    ConsoleLogger,
    LoggerList,
    TensorBoardLogger,
)
from rl4robot.common.training_loop import TrainingLoop

from mcmc_terrain_rl.mcmc_image_dict import make_mcmc_image_dict
from mcmc_terrain_rl.mcmc_terrain_env import McmcTerrainEnv

# 設定
# ================
env_id = "CustomTerrainAnt-v0"
device = "cpu"
actor_mlp_hidden_sizes = [64, 64]
value_mlp_hidden_sizes = [64, 64]
training_id = "mcmc"
out_dir = Path("out") / training_id
actor_critic_path = out_dir / "trained" / "actor_critic.pth"
# 訓練
training_steps = 2 ** 20  # 1M
tensorboard_dir = out_dir / "tb"
training_mcmc_image_dir = Path("terrain_images/training")
# 評価
evaluating_episodes = 100
evaluating_id = "mcmc"
evaluating_dir = out_dir / "eval" / evaluating_id
video_dir = evaluating_dir / "video"
result_json = evaluating_dir / "result.json"
num_videos = 10
evaluating_mcmc_image_dir = Path("terrain_images/evaluating")
# ================


# カリキュラム学習付きの訓練ループ
class CurriculumTrainingLoop(TrainingLoop):
    # _update()はPPOのモデル更新時（=horizonステップごと）に呼ばれる
    def _update(self):
        # 型チェック，違ったらassert（エラー）を投げる
        assert isinstance(self.env, McmcTerrainEnv)

        super()._update()

        # ==== MCMC地形のカリキュラム学習 ====
        # 注意！！
        # この実装はスコアに応じて適用的に j を変化させるものではなく、
        # 時刻に応じて j を変化させるもの

        progress = self.global_step / self.num_steps  # 訓練の進捗（0.0 〜 1.0）

        # TODO: 進捗に応じて j を決定する
        j = 0.5

        # j を変更
        # 次のエピソードからこの j が適用される
        self.env.set_j(j)

        # ================

    def _record_log(self):
        super()._record_log()

        self._record_curriculum_log()

    def _record_curriculum_log(self):
        if self.logger:
            assert isinstance(self.env, McmcTerrainEnv)

            self.logger.record("curriculum/j", self.env.j)


def backup_run_py(is_train: bool):
    import os
    import shutil

    backup_dir = out_dir if is_train else evaluating_dir
    os.makedirs(backup_dir)

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

    logger = LoggerList([ConsoleLogger(), TensorBoardLogger(tensorboard_dir)])

    # 環境を作成
    mcmc_image_dict = make_mcmc_image_dict(training_mcmc_image_dir)
    env = McmcTerrainEnv(env_id, mcmc_image_dict, 1.0)

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
    loop = CurriculumTrainingLoop(env, trainer, training_steps, logger=logger)
    loop.run()

    # 保存
    actor_critic.save_state(actor_critic_path)


def evaluate():
    backup_run_py(is_train=False)

    # 環境を作成
    mcmc_image_dict = make_mcmc_image_dict(evaluating_mcmc_image_dir)
    env = McmcTerrainEnv(env_id, mcmc_image_dict, 1.0)

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
    train()
    evaluate()
