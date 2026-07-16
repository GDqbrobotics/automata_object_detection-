from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from .camera import convert_depth_to_phys_coord_using_realsense

# Note: convert_depth_to_phys_coord_using_orbbec is imported lazily in segment_to_pose()
# to allow the app to work even when pyorbbecsdk is not available (e.g., RealSense mode)


def find_min_segment(cx: int, cy: int, contour: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    points = contour.squeeze(axis=1)
    translated_contour = points - [cx, cy]
    angles = np.arctan2(translated_contour[:, 1], translated_contour[:, 0])
    sorted_indices = np.argsort(angles)
    sorted_points = points[sorted_indices]
    sorted_angles = angles[sorted_indices]

    min_dist_sq = float("inf")
    result_points = ((0, 0), (0, 0))
    num_points = len(sorted_points)

    for i in range(num_points):
        p1 = sorted_points[i]
        angle1 = sorted_angles[i]
        target_angle = angle1 + np.pi
        if target_angle > np.pi:
            target_angle -= 2 * np.pi

        idx = np.searchsorted(sorted_angles, target_angle)
        idx1 = idx % num_points
        idx2 = (idx - 1 + num_points) % num_points

        diff1 = abs(target_angle - sorted_angles[idx1])
        diff2 = abs(target_angle - sorted_angles[idx2])
        best_idx = idx1 if diff1 < diff2 else idx2
        p2 = sorted_points[best_idx]

        dist_sq = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            result_points = (tuple(p1), tuple(p2))

    return result_points


def segment_to_pose(
    segment: Dict[str, Tuple[int, int]],
    depth: Any,
    coeff_height: float,
    coeff_width: float,
    camera_type: str = "realsense",
    depth_intrinsics: Any = None,
    extrinsic: Any = None,
) -> Optional[Dict[str, float]]:
    """Convert one grasp segment (full-frame pixel coordinates) to physical 3D coordinates.

    Args:
        segment: one detected grasp segment, as {"1": (x, y), "2": (x, y)},
            in full-frame pixels (the crop offset is already applied by the caller)
        depth: Depth image array
        coeff_height: Height coefficient for scaling
        coeff_width: Width coefficient for scaling
        camera_type: Type of camera ("realsense" or "orbbec")
        depth_intrinsics: Intrinsics for the depth camera (required for Orbbec)
        extrinsic: Extrinsic parameters for the camera (required for Orbbec)
    Returns:
        A dict with x_1, y_1, z_1, x_2, y_2, z_2, or None if the depth reading at
        either endpoint is invalid (0) and the pose cannot be estimated yet.
    """
    # Select appropriate coordinate conversion function
    coord_converter: Callable[[float, float, float, Any, Any], Tuple[float, float, float]]
    if camera_type == "orbbec":
        # Lazy import to avoid requiring pyorbbecsdk if not using Orbbec camera
        from .camera_orbbec import convert_depth_to_phys_coord_using_orbbec
        coord_converter = convert_depth_to_phys_coord_using_orbbec
    else:
        coord_converter = convert_depth_to_phys_coord_using_realsense

    x1 = segment["1"][0] * coeff_width
    y1 = segment["1"][1] * coeff_height
    z1 = float(depth[int(y1), int(x1)].item())
    z1, x1, y1 = coord_converter(x1, y1, z1, depth_intrinsics, extrinsic)

    x2 = segment["2"][0] * coeff_width
    y2 = segment["2"][1] * coeff_height
    z2 = float(depth[int(y2), int(x2)].item())
    z2, x2, y2 = coord_converter(x2, y2, z2, depth_intrinsics, extrinsic)

    if z1 == 0 or z2 == 0:
        return None

    return {"x_1": x1, "y_1": y1, "z_1": z1, "x_2": x2, "y_2": y2, "z_2": z2}
