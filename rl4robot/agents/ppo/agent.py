from typing import Final

import torch

from rl4robot.types import ActionArray, ObservationArray

from ..agent import Agent
from .actor_critic import ActorCritic

__all__ = [
    "PpoAgent",
]


class PpoAgent(Agent):
    actor_critic: Final[ActorCritic]

    def __init__(
        self,
        actor_critic: ActorCritic,
    ) -> None:
        self.actor_critic = actor_critic

        self.actor_critic.eval()

    def act(self, observation: ObservationArray) -> ActionArray:
        device = self.actor_critic.device

        with torch.no_grad():
            observation_ = torch.from_numpy(observation).to(device)
            action_ = self.actor_critic.get_actions(
                observation_, deterministic=True
            )
            action = action_.detach().cpu().numpy()

        return action
