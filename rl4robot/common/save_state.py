"""状態のファイル保存と読み込み可能なクラス"""


from typing import Protocol

from rl4robot.types import StrPath

__all__ = [
    "CanSaveState",
    "CanLoadState",
]


class CanSaveState(Protocol):
    """ファイルへ状態保存が可能なクラス"""

    def save_state(self, path: StrPath) -> None:
        """ファイルへ保存する。

        Parameters
        ----------
        path
            保存先ファイルパス
        """

        ...


class CanLoadState(Protocol):
    """ファイル保存が可能なクラス"""

    def load_state(self, path: StrPath) -> None:
        """ファイルから読み込む。

        Parameters
        ----------
        path
            読み込み先ファイルパス
        """

        ...
