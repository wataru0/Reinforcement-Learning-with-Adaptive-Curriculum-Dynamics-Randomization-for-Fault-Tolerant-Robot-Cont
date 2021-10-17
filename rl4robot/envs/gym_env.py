"""Gym環境のラッパー"""


from __future__ import annotations

from typing import Final, Optional

import gym

from rl4robot.types import ActionArray, ObservationArray, Range, RGBArray

from .env import Env, EnvSpec, EnvStep

__all__ = [
    "GymEnv",
]


class GymEnv(Env):
    """Gym環境のラッパー"""

    gym_env: Final[gym.wrappers.TimeLimit]
    _seed: Final[Optional[int]]

    def __init__(
        self, gym_env: gym.wrappers.TimeLimit, seed: Optional[int] = None
    ) -> None:
        """Gym環境ラッパー

        Parameters
        ----------
        gym_env
            Gym環境
        seed
            シード値
        """

        assert isinstance(gym_env.action_space, gym.spaces.Box)
        assert isinstance(gym_env.observation_space, gym.spaces.Box)

        self.gym_env = gym_env
        self._seed = seed

        self.gym_env.seed(seed)

    @property
    def unwrapped(self) -> Env:
        return self

    @property
    def spec(self) -> EnvSpec:
        id = self.gym_env.spec.id
        max_episode_steps = self.gym_env.spec.max_episode_steps
        gym_action_space = self.gym_env.action_space
        action_size = gym_action_space.shape[0]
        action_range = Range(gym_action_space.low, gym_action_space.high)
        gym_observation_space = self.gym_env.observation_space
        observation_size = gym_observation_space.shape[0]
        observation_range = Range(
            gym_observation_space.low, gym_observation_space.high
        )
        reward_range = Range(*self.gym_env.reward_range)

        return EnvSpec(
            id=id,
            max_episode_steps=max_episode_steps,
            action_size=action_size,
            action_range=action_range,
            observation_size=observation_size,
            observation_range=observation_range,
            reward_range=reward_range,
        )

    @property
    def seed(self) -> int:
        return self._seed

    def reset(self) -> ObservationArray:
        return self.gym_env.reset()

    def step(self, action: ActionArray) -> EnvStep:
        observation, reward, episode_done, _ = self.gym_env.step(action)

        return EnvStep(
            observation=observation, reward=reward, episode_done=episode_done
        )

    def render_image(
        self, *, width: int = 500, height: int = 500, **kwargs
    ) -> RGBArray:
        return self.gym_env.render(
            mode="rgb_array", width=width, height=height, **kwargs
        )

    def render_window(self, **kwargs) -> None:
        return self.gym_env.render(mode="human", **kwargs)

    def close(self) -> None:
        return self.gym_env.close()
