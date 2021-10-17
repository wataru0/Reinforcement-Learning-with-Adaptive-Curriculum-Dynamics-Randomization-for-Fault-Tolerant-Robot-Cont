"""PPOでの訓練"""


import statistics as stats
from dataclasses import dataclass
from typing import Final, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rl4robot.common.dataset import (
    DataLoader,
    Dataset,
    Minibatch,
    MinibatchIndex,
)
from rl4robot.common.gae import compute_advantage_hist
from rl4robot.common.loggers import Logger
from rl4robot.common.transition import (
    Transition,
    TransitionHist,
    TransitionHistStrage,
)
from rl4robot.envs import EnvStep
from rl4robot.types import (
    ActionArray,
    BatchActionArray,
    BatchActionTensor,
    BatchFloatArray,
    BatchFloatTensor,
    BatchObservationArray,
    BatchObservationTensor,
    ObservationArray,
)

from ..agent import Trainer
from .actor_critic import ActorCritic

__all__ = [
    "PpoTrainer",
]


@dataclass(frozen=True)
class _PpoMinibatch(Minibatch):
    observations_: BatchObservationTensor
    actions_: BatchActionTensor
    action_log_probs_: BatchFloatTensor
    advantages_: BatchFloatTensor
    returns_: BatchFloatTensor

    def __len__(self) -> int:
        return len(self.returns_)


@dataclass(frozen=True)
class _PpoDataset(Dataset[_PpoMinibatch]):
    observations: BatchObservationArray
    actions: BatchActionArray
    action_log_probs: BatchFloatArray
    advantages: BatchFloatArray
    returns: BatchFloatArray

    def __len__(self) -> int:
        return len(self.returns)

    def get_minibatch(
        self, minibatch_index: MinibatchIndex, device: torch.device
    ) -> _PpoMinibatch:
        def _to_mb_tensor(array: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(array[minibatch_index]).to(device)

        return _PpoMinibatch(
            observations_=_to_mb_tensor(self.observations),
            actions_=_to_mb_tensor(self.actions),
            action_log_probs_=_to_mb_tensor(self.action_log_probs),
            advantages_=_to_mb_tensor(self.advantages),
            returns_=_to_mb_tensor(self.returns),
        )


@dataclass(frozen=True)
class PpoLossInfo:
    loss: float
    loss_clip: float
    loss_value: float
    loss_entropy: float


def _calc_ppo_loss_mean(loss_infos: List[PpoLossInfo]) -> PpoLossInfo:
    loss_mean = stats.mean(loss_info.loss for loss_info in loss_infos)
    loss_clip_mean = stats.mean(
        loss_info.loss_clip for loss_info in loss_infos
    )
    loss_value_mean = stats.mean(
        loss_info.loss_value for loss_info in loss_infos
    )
    loss_entropy_mean = stats.mean(
        loss_info.loss_entropy for loss_info in loss_infos
    )

    return PpoLossInfo(
        loss=loss_mean,
        loss_clip=loss_clip_mean,
        loss_value=loss_value_mean,
        loss_entropy=loss_entropy_mean,
    )


class PpoUpdater:
    actor_critic: Final[ActorCritic]
    discount_gamma: Final[float]
    horizon: Final[int]
    num_epochs: Final[int]
    minibatch_size: Final[int]
    clip_epsilon: Final[float]
    coef_loss_value: Final[float]
    coef_loss_entropy: Final[float]
    max_grad_norm: Final[float]
    adam_lr: Final[float]
    gae_lambda: Final[float]
    data_loader: Final[DataLoader[_PpoMinibatch]]
    optimizer: Final[optim.Adam]

    def __init__(
        self,
        actor_critic: ActorCritic,
        seed: Optional[int] = None,
        *,
        discount_gamma: float = 0.99,
        horizon: int = 2048,
        num_epochs: int = 10,
        minibatch_size: int = 64,
        clip_epsilon: float = 0.2,
        coef_loss_value: float = 0.5,
        coef_loss_entropy: float = 0.0,
        max_grad_norm: float = 0.5,
        adam_lr: float = 3e-4,
        gae_lambda: float = 0.95,
    ) -> None:
        self.actor_critic = actor_critic

        self.horizon = horizon
        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size
        self.clip_epsilon = clip_epsilon
        self.coef_loss_value = coef_loss_value
        self.coef_loss_entropy = coef_loss_entropy
        self.max_grad_norm = max_grad_norm
        self.adam_lr = adam_lr
        self.gae_lambda = gae_lambda
        self.discount_gamma = discount_gamma

        self.data_loader = DataLoader(self.device, seed=seed)
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=self.adam_lr)

    @property
    def device(self) -> torch.device:
        return self.actor_critic.device

    def update(self, transition_hist: TransitionHist) -> PpoLossInfo:
        """モデルを更新する。

        Parameters
        ----------
        transition_hist
            遷移履歴

        Returns
        -------
        PpoLossInfo
            損失の情報
        """

        loss_infos: List[PpoLossInfo] = []

        dataset = self._make_dataset(transition_hist)
        self.data_loader.set_dataset(dataset)

        for _ in range(self.num_epochs):
            for minibatch in self.data_loader.get_minibatchs(
                self.minibatch_size, shuffle=True
            ):
                # 損失の計算
                loss_, loss_info = self._get_minibatch_loss(minibatch)

                loss_infos.append(loss_info)

                # 最適化
                self.optimizer.zero_grad()
                loss_.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

        return _calc_ppo_loss_mean(loss_infos)

    def _make_dataset(self, transition_hist: TransitionHist) -> _PpoDataset:
        observation_hist = transition_hist.observation
        observation_hist_ = torch.from_numpy(observation_hist).to(self.device)

        value_hist = (
            self.actor_critic.get_values(observation_hist_)
            .detach()
            .cpu()
            .numpy()
            .ravel()
        )

        next_observation_hist = transition_hist.next_observation
        next_observation_hist_ = torch.from_numpy(next_observation_hist).to(
            self.device
        )

        next_value_hist = (
            self.actor_critic.get_values(next_observation_hist_)
            .detach()
            .cpu()
            .numpy()
            .ravel()
        )

        next_reward_hist = transition_hist.next_reward
        next_episode_done_hist = transition_hist.next_episode_done

        advantage_hist = compute_advantage_hist(
            next_reward_hist,
            next_episode_done_hist,
            value_hist,
            next_value_hist,
            self.discount_gamma,
            self.gae_lambda,
        )

        return_hist = value_hist + advantage_hist

        action_hist = transition_hist.action
        action_hist_ = torch.from_numpy(action_hist).to(self.device)

        action_dist = self.actor_critic.get_action_dist(observation_hist_)
        action_log_prob_hist = (
            action_dist.log_prob(action_hist_)
            .sum(dim=-1)
            .detach()
            .cpu()
            .numpy()
        )

        return _PpoDataset(
            observations=observation_hist,
            actions=action_hist,
            action_log_probs=action_log_prob_hist,
            advantages=advantage_hist,
            returns=return_hist,
        )

    def _calc_loss(
        self,
        minibatch: _PpoMinibatch,
        values_: BatchFloatTensor,
        action_log_probs_: BatchFloatTensor,
        action_entropies_: BatchActionTensor,
        advantages_: BatchFloatTensor,
    ) -> Tuple[torch.Tensor, PpoLossInfo]:
        pi_ratio_ = torch.exp(
            action_log_probs_ - minibatch.action_log_probs_
        )  # π / π_old
        loss_clip_1_ = advantages_ * pi_ratio_
        loss_clip_2_ = advantages_ * torch.clamp(
            pi_ratio_, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
        )
        loss_clip_ = -torch.min(loss_clip_1_, loss_clip_2_).mean()

        loss_value_ = F.mse_loss(minibatch.returns_, values_)  # TD(λ)法を利用

        loss_entropy_ = -torch.mean(action_entropies_)

        loss_ = (
            loss_clip_
            + self.coef_loss_value * loss_value_
            + self.coef_loss_entropy * loss_entropy_
        )

        loss_info = PpoLossInfo(
            loss=float(loss_.item()),
            loss_clip=float(loss_clip_.item()),
            loss_value=float(loss_value_.item()),
            loss_entropy=float(loss_entropy_.item()),
        )

        return loss_, loss_info

    def _get_minibatch_loss(
        self, minibatch: _PpoMinibatch
    ) -> Tuple[torch.Tensor, PpoLossInfo]:
        values_ = self.actor_critic.get_values(
            minibatch.observations_
        ).flatten()

        action_dist = self.actor_critic.get_action_dist(
            minibatch.observations_
        )
        action_log_probs_ = action_dist.log_prob(minibatch.actions_).sum(
            dim=-1
        )
        action_entropies_ = action_dist.entropy().sum(dim=-1)

        # advantageを正規化
        advantages_ = (
            minibatch.advantages_ - minibatch.advantages_.mean()
        ) / (minibatch.advantages_.std() + 1e-8)

        return self._calc_loss(
            minibatch,
            values_,
            action_log_probs_,
            action_entropies_,
            advantages_,
        )


class PpoTrainer(Trainer):
    actor_critic: Final[ActorCritic]
    updater: Final[PpoUpdater]
    strage: Final[TransitionHistStrage]
    _action: Optional[ActionArray]
    _observation: Optional[ObservationArray]
    _loss_info: Optional[PpoLossInfo]

    def __init__(
        self,
        actor_critic: ActorCritic,
        seed: Optional[int] = None,
        *,
        discount_gamma: float = 0.99,
        horizon: int = 2048,
        num_epochs: int = 10,
        minibatch_size: int = 64,
        clip_epsilon: float = 0.2,
        coef_loss_value: float = 0.5,
        coef_loss_entropy: float = 0.0,
        max_grad_norm: float = 0.5,
        adam_lr: float = 3e-4,
        gae_lambda: float = 0.95,
    ) -> None:
        self.actor_critic = actor_critic
        self.updater = PpoUpdater(
            actor_critic,
            seed=seed,
            discount_gamma=discount_gamma,
            horizon=horizon,
            num_epochs=num_epochs,
            minibatch_size=minibatch_size,
            clip_epsilon=clip_epsilon,
            coef_loss_value=coef_loss_value,
            coef_loss_entropy=coef_loss_entropy,
            max_grad_norm=max_grad_norm,
            adam_lr=adam_lr,
            gae_lambda=gae_lambda,
        )
        self.strage = TransitionHistStrage(
            horizon,
            actor_critic.action_size,
            actor_critic.observation_size,
        )

        self._action = None
        self._observation = None
        self._loss_info = None

        self.actor_critic.train()

    def act(self, observation: ObservationArray) -> ActionArray:
        device = self.actor_critic.device

        with torch.no_grad():
            observation_ = torch.from_numpy(observation).to(device)
            action_ = self.actor_critic.get_actions(
                observation_, deterministic=False
            )
            action = action_.detach().cpu().numpy()

        self._action = action
        self._observation = observation

        return action

    def observe_result(self, env_step: EnvStep) -> None:
        assert self._action is not None and self._observation is not None

        self.strage.add(
            Transition(
                action=self._action,
                observation=self._observation,
                next_observation=env_step.observation,
                next_reward=env_step.reward,
                next_episode_done=env_step.episode_done,
            )
        )

        self._action = None
        self._observation = None

    def can_update(self) -> bool:
        return self.strage.is_full()

    def update(self) -> None:
        transition_hist = self.strage.transition_hist()
        self._loss_info = self.updater.update(transition_hist)

        self.strage.reset()

    def record_log(self, logger: Logger) -> None:
        if self._loss_info is None:
            return

        action_std_mean = float(self.actor_critic.action_std.mean())

        logger.record_from_dict(
            {
                "ppo/loss": self._loss_info.loss,
                "ppo/loss_clip": self._loss_info.loss_clip,
                "ppo/loss_value": self._loss_info.loss_value,
                "ppo/loss_entropy": self._loss_info.loss_entropy,
                "ppo/action_std": action_std_mean,
            }
        )

        self._loss_info = None
