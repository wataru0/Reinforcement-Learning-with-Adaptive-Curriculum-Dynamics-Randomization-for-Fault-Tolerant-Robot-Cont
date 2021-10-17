"""Generalized Advantage Estimation (GAE)
(https://arxiv.org/abs/1506.02438)

t でエピソード終了のとき
    δ_t = r_{t+1} - V(s_t)
    A_t = δ_t
それ以外
    δ_t = r_{t+1} + γ V(s_{t+1}) - V(s_t)
    A_t = δ_t + γ λ A_{t+1}
"""


import typing

import numpy as np

from rl4robot.types import BoolHistArray, FloatHistArray

__all__ = [
    "compute_vec_advantage_hist",
]


def compute_advantage_hist(
    next_reward_hist: FloatHistArray,
    next_episode_done_hist: BoolHistArray,
    value_hist: FloatHistArray,
    next_value_hist: FloatHistArray,
    discount_gamma: float,
    gae_lambda: float,
) -> FloatHistArray:
    horizon: int = value_hist.shape[0]
    next_episode_continue_hist = typing.cast(
        FloatHistArray,
        1.0 - next_episode_done_hist.astype(value_hist.dtype),
    )
    advantage_hist = typing.cast(FloatHistArray, np.empty_like(value_hist))

    # バッファの最後の次ステップのアドバンテージは不明なので 0 としておく
    last_advantage = 0.0
    for t in reversed(range(horizon)):
        delta = (
            next_reward_hist[t]
            + discount_gamma
            * next_value_hist[t]
            * next_episode_continue_hist[t]
            - value_hist[t]
        )
        last_advantage = (
            delta
            + discount_gamma
            * gae_lambda
            * last_advantage
            * next_episode_continue_hist[t]
        )
        advantage_hist[t] = last_advantage

    return advantage_hist
