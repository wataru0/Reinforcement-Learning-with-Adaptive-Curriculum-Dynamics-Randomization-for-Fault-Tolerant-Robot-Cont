"""ニューラルネットワークの層の重みを初期化"""


import torch.nn as nn

__all__ = [
    "orthogonal_init",
]


def orthogonal_init(
    module: nn.Linear, gain: float = 1.0, bias: float = 0.0
) -> nn.Linear:
    """全結合層の重みを直交初期化する

    Parameters
    ----------
    module
        モジュール
    gain
        重みの直交行列に掛ける係数
    bias
        定数バイアス

    Returns
    -------
    nn.Linear
        直行初期化された全結合層
    """

    nn.init.orthogonal_(module.weight, gain=gain)
    nn.init.constant_(module.bias, bias)

    return module
