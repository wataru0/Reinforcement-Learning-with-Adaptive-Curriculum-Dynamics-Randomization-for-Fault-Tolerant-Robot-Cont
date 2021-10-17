"""ロガー"""


from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Final, List, Protocol, Union

import torch.utils.tensorboard as tb

from rl4robot.types import StrPath

__all__ = [
    "LogKey",
    "LogValue",
    "LogKeyValues",
    "LoggerABC",
    "LoggerList",
    "ConsoleLogger",
    "TensorBoardLogger",
]


LogKey = str
LogValue = Union[int, float]
LogKeyValues = Dict[LogKey, LogValue]


class Logger(Protocol):
    """ログを記録するクラス"""

    @abstractmethod
    def record(self, key: LogKey, value: LogValue) -> None:
        """キーバリュ型のログをバッファに記憶する

        Parameters
        ----------
        key
            キー
        value
            値
        """

        raise NotImplementedError

    def record_from_dict(self, key_values: LogKeyValues) -> None:
        """辞書形式のログをバッファに記憶する

        Parameters
        ----------
        key_values
            キーと値の辞書
        """

        for key, value in key_values.items():
            self.record(key, value)

    @abstractmethod
    def dump(self, global_step: int) -> None:
        """バッファの内容を書き込む

        Parameters
        ----------
        global_step
            現在のタイムステップ
        """

        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """ロガーを閉じる"""

        raise NotImplementedError


class LoggerABC(ABC, Logger):
    key_values: LogKeyValues

    def __init__(self) -> None:
        super().__init__()

        self.key_values = {}

    def record(self, key: LogKey, value: LogValue) -> None:
        self.key_values[key] = value

    def dump(self, global_step: int) -> None:
        self._dump(global_step)

        self.key_values = {}

    def close(self):
        pass

    @abstractmethod
    def _dump(self, global_step: int) -> None:
        raise NotImplementedError


class LoggerList(LoggerABC):
    loggers: Final[List[Logger]]

    def __init__(self, loggers: List[Logger]):
        """複数のLoggerを同時に利用

        Parameters
        ----------
        loggers
            Loggerのリスト
        """

        super().__init__()

        self.loggers = loggers

    def close(self) -> None:
        for logger in self.loggers:
            logger.close()

    def _dump(self, global_step: int) -> None:
        for logger in self.loggers:
            logger.record_from_dict(self.key_values)
            logger.dump(global_step)


class ConsoleLogger(LoggerABC):
    kw: Final[int]
    vw: Final[int]

    def __init__(self, key_width: int = 25, value_width: int = 12) -> None:
        """コンソールにログを記録

        Parameters
        ----------
        key_width
            キーの表示幅
        value_width
            値の表示幅
        """

        super().__init__()

        self.kw = key_width
        self.vw = value_width

    def _dump(self, global_step: int) -> None:
        border = "-" * (self.kw + self.vw + 3)  # 3はマージン付きの縦線 ' | ' の幅

        print(border)
        for key, value in self.key_values.items():
            self._print_key_value(key, value)
        print(border)

    def _print_key_value(self, key: LogKey, value: LogValue) -> None:
        if isinstance(value, int):
            print(f"{key:<{self.kw}} | {value:>{self.vw},}")
        else:
            value_abs = abs(value)
            if value_abs > 1 or value_abs == 0.0:
                print(f"{key:<{self.kw}} | {value:>{self.vw}.2f}")
            elif value_abs > 1e-6:
                print(f"{key:<{self.kw}} | {value:>{self.vw}.8f}")
            else:
                print(f"{key:<{self.kw}} | {value:>{self.vw}.4E}")


class TensorBoardLogger(LoggerABC):
    directory: Final[Path]
    _writer: Final[tb.SummaryWriter]

    def __init__(self, directory: StrPath):
        """TensorBoardにログを記録

        Parameters
        ----------
        directory
            出力ディレクトリ
        """

        super().__init__()

        self.directory = Path(directory)
        self._writer = tb.SummaryWriter(directory)

    def close(self) -> None:
        self._writer.close()

    def _dump(self, global_step: int) -> None:
        for key, value in self.key_values.items():
            self._writer.add_scalar(key, value, global_step=global_step)
