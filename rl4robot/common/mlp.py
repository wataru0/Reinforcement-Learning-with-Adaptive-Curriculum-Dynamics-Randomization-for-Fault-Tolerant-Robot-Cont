"""多層パーセプトロン"""


import typing
from typing import Callable, Final, List, Optional

import torch
import torch.nn as nn

from rl4robot.common.activation import ActivationName, get_activation_module

__all__ = [
    "Mlp",
]


class Mlp(nn.Module):
    """多層パーセプトロン"""

    input_size: Final[int]
    output_size: Final[int]
    _seq: Final[nn.Sequential]

    def __init__(self, *modules: nn.Module) -> None:
        super().__init__()

        linears = typing.cast(
            List[nn.Linear],
            list(filter(lambda m: isinstance(m, nn.Linear), modules)),
        )

        self.input_size = linears[0].in_features
        self.output_size = linears[-1].out_features

        self._seq = nn.Sequential(*modules)

    @property
    def device(self) -> torch.device:
        params = list(self.parameters())
        return params[0].device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._seq(x)


_WeightInit = Callable[[nn.Linear], nn.Linear]

# 入力故障係数分増やしたい
def make_mlp(
    input_size: int,
    hidden_sizes: List[int],
    output_size: int,
    hidden_weight_init: Optional[_WeightInit] = None,
    output_weight_init: Optional[_WeightInit] = None,
    hidden_activation_name: Optional[ActivationName] = None,
    output_activation_name: Optional[ActivationName] = None,
) -> Mlp:
    """単純なMLPを作成する

    Parameters
    ----------
    input_size
        入力サイズ
    hidden_sizes
        隠れ層のユニット数のリスト
    output_size
        出力サイズ
    hidden_weight_init
        隠れ層の重み初期化関数
    output_weight_init
        出力層の重み初期化関数
    hidden_activation_name
        入力層の活性化関数
    output_activation_name
        出力層の活性化関数

    Returns
    -------
    Mlp
        多層パーセプトロン
    """

    if hidden_weight_init is None:
        hidden_weight_init = lambda m: m
    if output_weight_init is None:
        output_weight_init = lambda m: m

    modules: List[nn.Module] = []

    # 中間層まで
    last_size = input_size
    for hidden_size in hidden_sizes:
        linear = hidden_weight_init(nn.Linear(last_size, hidden_size))
        modules.append(linear)
        if hidden_activation_name is not None:
            activation = get_activation_module(hidden_activation_name)
            modules.append(activation)
        last_size = hidden_size

    # 最終層
    linear = output_weight_init(nn.Linear(last_size, output_size))
    modules.append(linear)
    if output_activation_name is not None:
        activation = get_activation_module(output_activation_name)
        modules.append(activation)

    return Mlp(*modules)
