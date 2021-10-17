"""ランダムな行動をとるエージェント"""


from __future__ import annotations

from typing import Final, Optional

import numpy as np

from rl4robot.types import ActionArray, ObservationArray, Range

from .agent import Agent


class RandomActionAgent(Agent):
    """ランダムな行動をとるエージェント"""

    action_range: Final[Range[ActionArray]]
    _random_state: Final[np.random.RandomState]

    def __init__(
        self, action_range: Range[ActionArray], seed: Optional[int] = None
    ):
        self.action_range = action_range
        self._random_state = np.random.RandomState(seed)

    def act(self, observation: ObservationArray) -> ActionArray:
        action_low, action_high = self.action_range
        return self._random_state.uniform(action_low, action_high)
