from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class CameraConfig:
    width: int = 1280
    height: int = 720
    K: Sequence[float] = (890.5523071289062, 0.0, 639.445068359375, 0.0, 890.5523071289062, 363.5865783691406, 0.0, 0.0, 1.0)
    D: Sequence[float] = (0.0, 0.0, 0.0, 0.0, 0.0)
    crop_width = 700
    crop_height = 540
    crop_starting_row = int((1080 - int(crop_height))/2)
    crop_starting_col = int((1920 - int(crop_width))/2)


DEFAULT_CAMERA_CONFIG = CameraConfig()
