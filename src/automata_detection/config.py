from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class CameraConfig:
    width: int = 1280
    height: int = 720
    K: Sequence[float] = (890.5523071289062, 0.0, 639.445068359375, 0.0, 890.5523071289062, 363.5865783691406, 0.0, 0.0, 1.0)
    D: Sequence[float] = (0.0, 0.0, 0.0, 0.0, 0.0)
    # Crop region (in full-frame pixels) that BiRefNet runs on. It must cover the
    # whole grid so every fragment is segmented. Tune these 4 values while looking
    # at result.png until the entire grid fits inside the crop.
    crop_width = 720
    crop_height = 420
    crop_starting_row = 220
    crop_starting_col = 210


DEFAULT_CAMERA_CONFIG = CameraConfig()


@dataclass(frozen=True)
class ArucoConfig:
    dictionary: str = "DICT_5X5_250"      # ArUco dictionary name (cv2.aruco.<name>)
    marker_length_m: float = 0.014        # measured marker side length in meters
    marker_separation_m: float = 0.0014   # measured gap between markers in meters
    markers_x: int = 2                    # number of markers along X (grid board)
    markers_y: int = 2                    # number of markers along Y (grid board)


DEFAULT_ARUCO_CONFIG = ArucoConfig()


@dataclass(frozen=True)
class GridConfig:
    n_cols: int = 10              # number of columns (cells along X, labels 1..10)
    n_rows: int = 6               # number of rows (cells along Y, labels A..F)
    cell_size_mm: float = 50.0    # size of each square cell in millimeters
    occupancy_threshold: float = 0.05  # a cell counts as occupied if a fragment covers at least this fraction of it (0..1)
    coverage_samples: int = 5     # NxN sample points per cell used to estimate the covered fraction


DEFAULT_GRID_CONFIG = GridConfig()
