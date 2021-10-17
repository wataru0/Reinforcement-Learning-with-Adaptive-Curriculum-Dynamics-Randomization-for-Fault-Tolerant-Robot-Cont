import math
from pathlib import Path
from typing import Final, List

import torch
import os
import torch.distributions as Dist
import torch.nn as nn

from rl4robot.common.mlp import Mlp, make_mlp
from rl4robot.common.save_state import CanLoadState, CanSaveState
from rl4robot.common.weight_init import orthogonal_init
from rl4robot.types import (
    ActionArray,
    BatchActionTensor,
    BatchFloatTensor,
    BatchObservationTensor,
    StrPath,
)

__all__ = [
    "ActorCritic",
]


class ActorCritic(nn.Module, CanSaveState, CanLoadState):
    observation_size: Final[int]
    action_size: Final[int]
    _actor_mlp: Final[Mlp]
    _value_mlp: Final[Mlp]
    _action_std_: Final[nn.Parameter]

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        actor_mlp_hidden_sizes: List[int],
        value_mlp_hidden_sizes: List[int],
    ) -> None:
        """Actor-Critic型の方策ネットワーク

        Parameters
        ----------
        observation_size
            観測値の次元
        action_size
            行動の次元
        actor_mlp_hidden_sizes
            アクターMLPの各隠れ層のユニット数
        value_mlp_hidden_sizes
            値関数MLPの各隠れ層のユニット数
        """

        super().__init__()

        self.observation_size = observation_size
        self.action_size = action_size

        self._actor_mlp = ActorCritic._make_actor_mlp(
            observation_size, action_size, actor_mlp_hidden_sizes
        )
        self._value_mlp = ActorCritic._make_value_mlp(
            observation_size, value_mlp_hidden_sizes
        )

        self._action_std_ = nn.Parameter(
            torch.ones(action_size), requires_grad=True
        )

    @property
    def device(self) -> torch.device:
        params = list(self.parameters())
        return params[0].device

    @property
    def action_std(self) -> ActionArray:
        return self._action_std_.detach().cpu().numpy()

    def get_action_dist(
        self, observations_: BatchObservationTensor
    ) -> Dist.Normal:
        action_means_ = self._actor_mlp(observations_)
        action_stds_ = torch.ones_like(action_means_) * self._action_std_
        return Dist.Normal(action_means_, action_stds_)

    def get_actions(
        self,
        observations_: BatchObservationTensor,
        deterministic=False,
    ) -> BatchActionTensor:
        action_dist = self.get_action_dist(observations_)

        if deterministic:
            actions_ = action_dist.mean
        else:
            actions_ = action_dist.rsample()

        return actions_

    def get_values(
        self, observations_: BatchObservationTensor
    ) -> BatchFloatTensor:
        return self._value_mlp(observations_)

    def save_state(self, path: StrPath) -> None:
        path = Path(path)
        os.makedirs(path.parent, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_state(self, path: StrPath) -> None:
        self.load_state_dict(torch.load(path))

    @staticmethod
    def _make_actor_mlp(
        observation_size: int, action_size: int, hidden_sizes: List[int]
    ) -> Mlp:
        return make_mlp(
            observation_size,
            hidden_sizes,
            action_size,
            hidden_activation_name="tanh",
            hidden_weight_init=lambda m: orthogonal_init(m, gain=math.sqrt(2)),
            output_weight_init=lambda m: orthogonal_init(m, gain=0.01),
        )

    @staticmethod
    def _make_value_mlp(observation_size: int, hidden_sizes: List[int]) -> Mlp:
        return make_mlp(
            observation_size,
            hidden_sizes,
            1,
            hidden_activation_name="tanh",
            hidden_weight_init=lambda m: orthogonal_init(m, gain=math.sqrt(2)),
            output_weight_init=lambda m: orthogonal_init(m, gain=1.0),
        )
