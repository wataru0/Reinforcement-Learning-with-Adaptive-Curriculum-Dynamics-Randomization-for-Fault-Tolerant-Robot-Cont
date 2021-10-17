"""遷移とその履歴"""


from dataclasses import dataclass
from typing import Final

import numpy as np

from rl4robot.types import (
    ActionArray,
    ActionHistArray,
    BoolHistArray,
    FloatHistArray,
    ObservationArray,
    ObservationHistArray,
)

__all__ = [
    "Transition",
    "TransitionHist",
    "TransitionHistStrage",
]


@dataclass(frozen=True)
class Transition:
    action: ActionArray
    observation: ObservationArray
    next_observation: ObservationArray
    next_reward: float
    next_episode_done: bool

    @property
    def action_size(self) -> int:
        return self.action.shape[-1]

    @property
    def observation_size(self) -> int:
        return self.observation.shape[-1]


@dataclass(frozen=True)
class TransitionHist:
    action: ActionHistArray
    observation: ObservationHistArray
    next_observation: ObservationHistArray
    next_reward: FloatHistArray
    next_episode_done: BoolHistArray

    @property
    def horizon(self) -> int:
        return self.next_episode_done.shape[0]

    @property
    def action_size(self) -> int:
        return self.action.shape[-1]

    @property
    def observation_size(self) -> int:
        return self.observation.shape[-1]


class TransitionHistStrage:
    max_horizon: Final[int]
    horizon: int
    action_size: Final[int]
    observation_size: Final[int]
    _action_hist: Final[ActionHistArray]
    _observation_hist: Final[ActionHistArray]
    _next_observation_hist: Final[ObservationHistArray]
    _next_reward_hist: Final[FloatHistArray]
    _next_episode_done_hist: Final[BoolHistArray]

    def __init__(
        self,
        max_horizon: int,
        action_size: int,
        observation_size: int,
    ) -> None:
        self.max_horizon = max_horizon
        self.action_size = action_size
        self.observation_size = observation_size

        self.horizon = 0

        # 高速化のために予めメモリを確保
        self._action_hist = np.empty((max_horizon, action_size), dtype=float)
        self._observation_hist = np.empty(
            (max_horizon, observation_size), dtype=float
        )
        self._next_observation_hist = np.empty(
            (max_horizon, observation_size), dtype=float
        )
        self._next_reward_hist = np.empty((max_horizon,), dtype=float)
        self._next_episode_done_hist = np.empty((max_horizon,), dtype=bool)

    def is_full(self) -> bool:
        return self.horizon == self.max_horizon

    def reset(self) -> None:
        self.horizon = 0

    def add(self, transition: Transition) -> None:
        assert not self.is_full()

        self._action_hist[self.horizon] = transition.action.copy()
        self._observation_hist[self.horizon] = transition.observation.copy()
        self._next_observation_hist[
            self.horizon
        ] = transition.next_observation.copy()
        self._next_reward_hist[self.horizon] = transition.next_reward
        self._next_episode_done_hist[
            self.horizon
        ] = transition.next_episode_done

        self.horizon += 1

    def action_hist(self) -> ActionHistArray:
        return self._action_hist[: self.horizon]

    def observation_hist(self) -> ActionHistArray:
        return self._observation_hist[: self.horizon]

    def next_observation_hist(self) -> ActionHistArray:
        return self._next_observation_hist[: self.horizon]

    def next_reward_hist(self) -> FloatHistArray:
        return self._next_reward_hist[: self.horizon]

    def next_episode_done_hist(self) -> BoolHistArray:
        return self._next_episode_done_hist[: self.horizon]

    def transition_hist(self) -> TransitionHist:
        return TransitionHist(
            action=self.action_hist(),
            observation=self.observation_hist(),
            next_observation=self.next_observation_hist(),
            next_reward=self.next_reward_hist(),
            next_episode_done=self.next_episode_done_hist(),
        )
