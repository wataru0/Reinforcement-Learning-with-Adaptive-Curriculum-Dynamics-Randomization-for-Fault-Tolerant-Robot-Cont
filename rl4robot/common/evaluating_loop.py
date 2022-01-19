import statistics as stats
from dataclasses import dataclass
from pathlib import Path
from typing import Final, List, Optional, Tuple

import numpy as np
import tqdm

from rl4robot.agents import Agent
from rl4robot.common.video_writer import VideoWriter
from rl4robot.envs import Env
from rl4robot.types import StrPath


@dataclass(frozen=True)
class EvaluatingResult:
    episode_lengths: List[int]
    episode_returns: List[float]
    episode_forward_rewards: List[float]

    def episode_length_mean(self) -> float:
        return stats.mean(self.episode_lengths)

    def episode_length_std(self) -> float:
        return stats.stdev(self.episode_lengths)

    def episode_return_mean(self) -> float:
        return stats.mean(self.episode_returns)

    def episode_return_std(self) -> float:
        return stats.stdev(self.episode_returns)
    
    def episode_forward_reward_mean(self) -> float:
        return stats.mean(self.episode_forward_rewards)

    def episode_forward_reward_std(self) -> float:
        return stats.stdev(self.episode_forward_rewards)


class EvaluatingLoop:
    env: Final[Env]
    agent: Final[Agent]
    num_episodes: Final[int]
    video_dir: Final[Optional[Path]]
    num_videos: Final[int]
    gene_flag: Final[bool]

    def __init__(
        self,
        env: Env,
        agent: Agent,
        num_episodes: int,
        video_dir: Optional[StrPath] = None,
        num_videos: int = 0,
        gene_flag: bool = False,
    ) -> None:
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes
        self.video_dir = Path(video_dir) if video_dir else None
        self.num_videos = num_videos
        self.gene_flag = gene_flag

    def run(self) -> EvaluatingResult:
        """評価ループを実行する。"""

        episode_lengths: List[int] = []
        episode_returns: List[float] = []
        episode_forward_rewards: List[float] = []

        if self.gene_flag:
            for i in range(self.num_episodes):
                video_path = (
                self.video_dir / f"episode-{i + 1:03}.mp4"
                if i < self.num_videos
                else None
            )

            episode_length, episode_return, episode_forward_reward = self._collect_episode(video_path)

            episode_lengths.append(episode_length)
            episode_returns.append(episode_return)
            episode_forward_rewards.append(episode_forward_reward)

            return EvaluatingResult(
                episode_lengths=episode_lengths, episode_returns=episode_returns, episode_forward_rewards=episode_forward_rewards
            )
        else:
            for i in tqdm.tqdm(range(self.num_episodes)):
                video_path = (
                    self.video_dir / f"episode-{i + 1:03}.mp4"
                    if i < self.num_videos
                    else None
                )

                episode_length, episode_return, episode_forward_reward = self._collect_episode(video_path)

                episode_lengths.append(episode_length)
                episode_returns.append(episode_return)
                episode_forward_rewards.append(episode_forward_reward)

            return EvaluatingResult(
                episode_lengths=episode_lengths, episode_returns=episode_returns, episode_forward_rewards=episode_forward_rewards
            )

    def _collect_episode(
        self, video_path: Optional[StrPath] = None
    ) -> Tuple[int, float]:
        """１エピソード評価する。

        Returns
        -------
        Tuple[int, float]
            エピソード長とエピソード収益
        """

        video_writer = VideoWriter(video_path) if video_path else None

        action_low, action_high = self.env.spec.action_range
        episode_done = False
        episode_length = 0
        episode_return = 0.0
        episode_forward_reward = 0.0

        observation = self.env.reset()

        while True:
            if video_writer:
                rgb_array = self.env.render_image()
                video_writer.write(rgb_array)
            if episode_done:
                break
            action = self.agent.act(observation)
            action = np.clip(action, action_low, action_high)
            env_step = self.env.step(action)

            observation = env_step.observation
            episode_done = env_step.episode_done
            episode_length += 1
            episode_return += env_step.reward
            episode_forward_reward += env_step.info['reward_forward']

        if video_writer:
            video_writer.close()

        return episode_length, episode_return, episode_forward_reward
