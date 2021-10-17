"""型を定義"""


from os import PathLike
from typing import Generic, NamedTuple, TypeVar, Union

import numpy as np
import torch

__all__ = [
    # Array
    "ActionArray",
    "ObservationArray",
    # BatchArray
    # HistArray
    "BoolHistArray",
    "IntHistArray",
    "FloatHistArray",
    "ActionHistArray",
    "ObservationHistArray",
    # 範囲
    "Range",
    # パス
    "StrPath",
]


# ======================================
# Array
# ======================================

# shape=(ac_size, ), dtype=float
ActionArray = np.ndarray

# shape=(ob_size, ), dtype=float
ObservationArray = np.ndarray


# ======================================
# Tensor
# ======================================

# size=(ac_size, ), dtype=torch.float64
ActionArray = torch.Tensor

# size=(ob_size, ), dtype=torch.float64
ObservationArray = torch.Tensor


# ======================================
# HistArray
# ======================================

# shape=(horizon, ), dtype=bool
BoolHistArray = np.ndarray

# shape=(horizon, ), dtype=int
IntHistArray = np.ndarray

# shape=(horizon, ), dtype=float
FloatHistArray = np.ndarray

# shape=(horizon, ac_size, ), dtype=float
ActionHistArray = np.ndarray

# shape=(horizon, ob_size, ), dtype=float
ObservationHistArray = np.ndarray


# ======================================
# BatchArray
# ======================================

# shape=(??, ), dtype=bool
BatchBoolArray = np.ndarray

# shape=(??, ), dtype=int
BatchIntArray = np.ndarray

# shape=(??, ), dtype=float
BatchFloatArray = np.ndarray

# shape=(??, ac_size, ), dtype=float
BatchActionArray = np.ndarray

# shape=(??, ob_size, ), dtype=float
BatchObservationArray = np.ndarray


# ======================================
# BatchTensor
# ======================================

# size=(??, ), dtype=bool
BatchBoolTensor = torch.Tensor

# size=(??, ), dtype=int
BatchIntTensor = torch.Tensor

# size=(??, ), dtype=torch.float64
BatchFloatTensor = torch.Tensor

# size=(??, ac_size, ), dtype=torch.float64
BatchActionTensor = torch.Tensor

# size=(??, ob_size, ), dtype=torch.float64
BatchObservationTensor = torch.Tensor


# ======================================
# 画像
# ======================================

# shape=(height, width, 3), dtype=numpy.uint8
RGBArray = np.ndarray


# ======================================
# 範囲
# ======================================

_T = TypeVar("_T")

# NOTE
# Rangeを利用するファイルでは
# `from __future__ import annotations`
# を記述すること。
# https://stackoverflow.com/questions/50530959/generic=namedtuple=in=python=3=6
class Range(NamedTuple, Generic[_T]):
    low: _T
    high: _T


# ======================================
# パス
# ======================================

StrPath = Union[str, PathLike]
