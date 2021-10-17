"""環境ラッパーのデフォルト実装"""


from typing import Optional

from rl4robot.types import ActionArray, ObservationArray, RGBArray

from .env import Env, EnvSpec, EnvStep

__all__ = [
    "EnvWrapper",
]


class EnvWrapper(Env):
    """環境ラッパーのデフォルト実装"""

    env: Env

    def __init__(self, env: Env) -> None:
        """環境ラッパーのデフォルト実装

        Parameters
        ----------
        env
            環境
        """

        self.env = env

    def __str__(self) -> str:
        return f"{type(self).__name__}<{self.env}>"

    @property
    def unwrapped(self) -> Env:
        return self.env.unwrapped

    @property
    def spec(self) -> EnvSpec:
        return self.env.spec

    @property
    def seed(self) -> Optional[int]:
        return self.env.seed

    def reset(self) -> ObservationArray:
        return self.env.reset()

    def step(self, action: ActionArray) -> EnvStep:
        return self.env.step(action)

    def render_image(
        self,
        *,
        width: int = 500,
        height: int = 500,
        blackout: bool = False,
        **kwargs,
    ) -> RGBArray:
        return self.env.render_image(width=width, height=height, **kwargs)

    def render_window(self, **kwargs) -> None:
        return self.env.render_window(**kwargs)

    def close(self) -> None:
        return self.env.close()
