"""データセット"""


from typing import Final, Generic, Iterator, Optional, Protocol, TypeVar

import numpy as np
import torch

__all__ = [
    "MinibatchIndex",
    "EpochMinibatchIndex",
    "Minibatch",
    "Dataset",
    "DataLoader",
]


MinibatchIndex = np.ndarray  # shape=(mb_size, ), dtype=int
EpochMinibatchIndex = np.ndarray  # shape=(iter_size, mb_size), dtype=int


class Minibatch(Protocol):
    """ミニバッチサンプル"""

    def __len__(self) -> int:
        ...


_Minibatch = TypeVar("_Minibatch", bound=Minibatch)


class Dataset(Protocol[_Minibatch]):
    """データセット"""

    def __len__(self) -> int:
        ...

    def get_minibatch(
        self, minibatch_index: MinibatchIndex, device: torch.device
    ) -> _Minibatch:
        ...


class DataLoader(Generic[_Minibatch]):
    """データローダ"""

    device: Final[torch.device]
    _random_state: Final[np.random.RandomState]
    _dataset: Optional[Dataset[_Minibatch]]

    def __init__(
        self, device: torch.device, seed: Optional[int] = None
    ) -> None:
        self.device = torch.device(device)
        self._random_state = np.random.RandomState(seed)
        self._dataset = None

    def __len__(self) -> int:
        if self._dataset is None:
            return 0
        return len(self._dataset)

    def set_dataset(self, dataset: Optional[Dataset[_Minibatch]]) -> None:
        """データセットを登録

        Parameters
        ----------
        dataset
            データセット
        """

        self._dataset = dataset

    def get_minibatchs(
        self, minibatch_size: int, shuffle: bool = False
    ) -> Iterator[_Minibatch]:
        """データセット内のサンプルをミニバッチごとに取得

        Parameters
        ----------
        minibatch_size
            ミニバッチ内のサンプル数
        shuffle
            サンプルの順番をシャッフルをするか

        Yields
        -------
        Iterator[_Minibatch]
            ミニバッチサンプルのイテレータ
        """

        epoch_mb_index = self._generate_epoch_minibatch_index(
            minibatch_size, shuffle=shuffle
        )

        for mb_index in epoch_mb_index[:]:
            mb = self._dataset.get_minibatch(mb_index, self.device)
            yield mb

    def _generate_epoch_minibatch_index(
        self, minibatch_size: int, shuffle: bool = False
    ) -> EpochMinibatchIndex:
        """1エポック分のミニバッチを生成ためのインデックスを生成

        Parameters
        ----------
        minibatch_size
            ミニバッチ内のサンプル数
        shuffle
            サンプルの順番をシャッフルをするか

        Returns
        -------
        EpochMinibatchIndex
            [description]
        """

        assert len(self) % minibatch_size == 0

        if shuffle:
            epoch_mb_index = self._random_state.permutation(len(self))
        else:
            epoch_mb_index = np.arange(len(self))

        return epoch_mb_index.reshape((-1, minibatch_size))
