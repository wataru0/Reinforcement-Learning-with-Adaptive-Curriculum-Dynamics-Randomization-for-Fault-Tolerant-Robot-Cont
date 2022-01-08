"""環境"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from rl4robot.types import ActionArray, ObservationArray, Range, RGBArray

__all__ = [
    "EnvSpec",
    "EnvStep",
    "Env",
]


@dataclass
class EnvSpec:
    """環境に関する情報"""

    id: str  # 環境ID
    action_size: int  # 行動の次元数
    action_range: Range[ActionArray]  # 行動の範囲
    observation_size: int  # 観測値の次元数
    observation_range: Range[ObservationArray]  # 観測値の範囲
    reward_range: Range[float]  # 報酬の範囲
    max_episode_steps: int  # 最大エピソード長


@dataclass
class EnvStep:
    """環境の新しい状況"""

    observation: ObservationArray  # 観測値
    reward: float  # 報酬
    episode_done: bool  # エピソードが終了したか
    info: dict # infomation


class Env(Protocol):
    """環境"""

    @property
    def unwrapped(self) -> Env:
        """Envプロトコルを満たす最も内側の環境"""

        ...

    @property
    def spec(self) -> EnvSpec:
        """環境に関する情報"""

        ...

    @property
    def seed(self) -> Optional[int]:
        """シード値"""

        ...

    def reset(self) -> ObservationArray:
        """環境をリセットする。

        Returns
        -------
        ObservationArray
            観測値
        """

        ...

    def step(self, action: ActionArray) -> EnvStep:
        """環境を１ステップすすめる。

        Parameters
        ----------
        action
            行動

        Returns
        -------
        StepSpec
            環境の新しい状況
        """

        ...

    def render_image(
        self, *, width: int = 500, height: int = 500, **kwargs
    ) -> RGBArray:
        """画像を描画する。

        Parameters
        ----------
        width
            画像の幅[px]
        height
            画像の高さ[px]

        Returns
        -------
        RGBArray
            画像のピクセル値
        """

        ...

    def render_window(self, **kwargs) -> None:
        """ウィンドウを描画する。"""

        ...

    def close(self) -> None:
        """環境を終了する。"""

        ...
