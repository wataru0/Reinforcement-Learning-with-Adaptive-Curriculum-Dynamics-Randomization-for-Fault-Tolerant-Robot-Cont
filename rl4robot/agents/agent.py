from typing import Optional, Protocol

from rl4robot.common.loggers import Logger
from rl4robot.envs import EnvStep
from rl4robot.types import ActionArray, ObservationArray

__all__ = [
    "Agent",
]


class Agent(Protocol):
    def act(self, observation: ObservationArray) -> ActionArray:
        """行動を決定"""

        ...


class Trainer(Agent, Protocol):
    def observe_result(self, env_step: EnvStep) -> None:
        ...

    def can_update(self) -> bool:
        ...

    def update(self) -> None:
        ...

    def record_log(self, logger: Logger) -> None:
        ...
