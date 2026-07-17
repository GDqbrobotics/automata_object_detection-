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


@dataclass(frozen=True)
class TrackerConfig:
    match_distance_px: float = 50.0  # max centroid movement (pixels) between cycles to still count as the same artifact
    miss_limit: int = 15             # consecutive inference cycles an artifact can go undetected before it is removed
    heartbeat_s: float = 5.0         # republish the full artifact list at least this often, even with no changes
    reacquire_window_s: float = 20.0  # a removed artifact can get its old id back if a detection reappears at the same spot within this time
    confirm_hits: int = 3            # consecutive detections a new artifact needs before it gets an id (kills one-cycle BiRefNet phantoms)


DEFAULT_TRACKER_CONFIG = TrackerConfig()


@dataclass(frozen=True)
class DetectionConfig:
    alpha_threshold: int = 250    # BiRefNet mask pixels with alpha >= this count as object (phantom masks are usually weaker)
    min_contour_size: int = 200   # contours smaller than this are noise and are ignored
    max_contour_size: int = 4000  # contours bigger than this are ignored (oversized blobs)
    max_area_fraction: float = 0.3  # a blob covering more than this fraction of the crop is the bare table (mat removed), not a fragment


DEFAULT_DETECTION_CONFIG = DetectionConfig()


@dataclass(frozen=True)
class DepthFilterConfig:
    max_artifact_height_mm: float = 40.0    # a blob sticking out of the table more than this is a hand/arm, not a fragment
    min_artifact_height_mm: float = 2.0     # at confirmation, a blob flatter than this (vs its local surroundings) is mat texture, not a fragment; 0 disables
    occlusion_freeze_fraction: float = 0.10  # if more than this fraction of the crop is "tall", the scene is occluded
    table_alpha: float = 0.02               # slow EMA used to keep the table depth estimate up to date
    table_relearn_cycles: int = 30          # after this many cycles of steady disagreement, re-learn the table depth from scratch


DEFAULT_DEPTH_FILTER_CONFIG = DepthFilterConfig()

# One color per artifact (BGR), chosen by its stable id (id % len(colors)), cycled
# if there are more artifacts than colors. Shared between result.png (inference)
# and the ArUco grid overlay, so a given artifact has the same color everywhere.
ARTIFACT_COLORS = [
    (255, 0, 0),      # blue
    (0, 0, 255),      # red
    (0, 200, 0),      # green
    (0, 200, 255),    # yellow
    (255, 0, 255),    # magenta
    (255, 200, 0),    # cyan
    (0, 128, 255),    # orange
]
