import os
from pathlib import Path
from typing import Any, Final

import cv2
import numpy as np

from rl4robot.types import RGBArray, StrPath

_suffix_fourcc_dict = {
    ".mp4": "mp4v",
    ".mov": "mp4v",
}


def _get_fourcc(path: Path) -> Any:
    suffix = path.suffix

    if suffix not in _suffix_fourcc_dict:
        raise ValueError

    return cv2.VideoWriter_fourcc(*_suffix_fourcc_dict[suffix])


class VideoWriter:
    writer: Final[cv2.VideoWriter]

    def __init__(
        self, path: StrPath, *, width=500, height=500, fps=50
    ) -> None:
        path = Path(path)
        fourcc = _get_fourcc(path)

        os.makedirs(path.parent, exist_ok=True)
        self.writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    def __del__(self) -> None:
        try:
            self.close()
        except:
            pass

    def write(self, rgb_array: RGBArray) -> None:
        bgr_array = np.flip(rgb_array, axis=-1)  # OpenCVはBGR画像を扱う
        self.writer.write(bgr_array)

    def close(self) -> None:
        return self.writer.release()
