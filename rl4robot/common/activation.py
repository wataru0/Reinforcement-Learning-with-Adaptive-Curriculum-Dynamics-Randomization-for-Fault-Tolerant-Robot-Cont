from typing import Dict, Literal, Type, Union

import torch.nn as nn

__all__ = [
    "ActivationName",
    "ActivationModule",
    "get_activation_module",
]


ActivationName = Literal[
    "sigmoid",
    "tanh",
    "softplus",
    "relu",
]


ActivationModule = Union[
    nn.Sigmoid,
    nn.Tanh,
    nn.Softplus,
    nn.ReLU,
]


_activation_dict: Dict[ActivationName, Type[ActivationModule]] = {
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "softplus": nn.Softplus,
    "relu": nn.ReLU,
}


def get_activation_module(name: ActivationName) -> ActivationModule:
    """活性化関数のモジュールを取得"""

    activation_module_type = _activation_dict.get(name)

    if activation_module_type is None:
        raise ValueError

    activation_module = activation_module_type()
    return activation_module
