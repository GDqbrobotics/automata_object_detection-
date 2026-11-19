import time
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from .camera import DEFAULT_CAMERA_CONFIG, ImageCropper, convert_depth_to_phys_coord_using_realsense
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for matplotlib
import matplotlib.pyplot as plt

# Note: convert_depth_to_phys_coord_using_orbbec is imported lazily in pixel2pose()
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
    
    # Increase the distance between the two points by 10% to ensure they are not too close
    p1, p2 = result_points
    mid_point = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
    vector = (p2[0] - p1[0], p2[1] - p1[1])
    new_p1 = (int(mid_point[0] - vector[0] * 0.55), int(mid_point[1] - vector[1] * 0.55))
    new_p2 = (int(mid_point[0] + vector[0] * 0.55), int(mid_point[1] + vector[1] * 0.55))
    result_points = (new_p1, new_p2)   
    return result_points


def pixel2pose(
    results: List[Dict[str, Tuple[int, int]]],
    depth: Any,
    coeff_height: float,
    coeff_width: float,
    camera_type: str = "realsense",
    depth_intrinsics: Any = None,
    extrinsic: Any = None,
) -> List[Dict[str, float]]:
    """Convert pixel coordinates to physical 3D coordinates.
    
    Args:
        results: List of detected contours with segment points
        depth: Depth image array
        coeff_height: Height coefficient for scaling
        coeff_width: Width coefficient for scaling
        camera_type: Type of camera ("realsense" or "orbbec")
        depth_intrinsics: Intrinsics for the depth camera (required for Orbbec)
        extrinsic: Extrinsic parameters for the camera (required for Orbbec)
    Returns:
        List of pose messages with 3D coordinates
    """
    # Select appropriate coordinate conversion function
    coord_converter: Callable[[float, float, float, Any, Any], Tuple[float, float, float]]
    if camera_type == "orbbec":
        # Lazy import to avoid requiring pyorbbecsdk if not using Orbbec camera
        from .camera_orbbec import convert_depth_to_phys_coord_using_orbbec
        coord_converter = convert_depth_to_phys_coord_using_orbbec
    else:
        coord_converter = convert_depth_to_phys_coord_using_realsense

    message: List[Dict[str, float]] = []
    cropper = ImageCropper(
        DEFAULT_CAMERA_CONFIG.crop_width,
        DEFAULT_CAMERA_CONFIG.crop_height,
        DEFAULT_CAMERA_CONFIG.crop_starting_row,
        DEFAULT_CAMERA_CONFIG.crop_starting_col,
    )

    for index, result in enumerate(results):
        y1, x1 = cropper.cropped2orig(result["1"][1], result["1"][0])
        x1 = x1 * coeff_width
        y1 = y1 * coeff_height
        z1 = float(depth[int(y1), int(x1)].item())
        z1, x1, y1 = coord_converter(x1, y1, z1, depth_intrinsics, extrinsic)

        y2, x2 = cropper.cropped2orig(result["2"][1], result["2"][0])
        x2 = x2 * coeff_width
        y2 = y2 * coeff_height
        z2 = float(depth[int(y2), int(x2)].item())
        z2, x2, y2 = coord_converter(x2, y2, z2, depth_intrinsics, extrinsic)

        if False:  # Debugging: Save depth values as heatmap
            
            frame_y_min, frame_x_min = cropper.cropped2orig(0, 0)
            frame_y_max, frame_x_max = cropper.cropped2orig(DEFAULT_CAMERA_CONFIG.crop_height, DEFAULT_CAMERA_CONFIG.crop_width)
            target_x_min = int(frame_x_min * coeff_width)
            target_x_max = int(frame_x_max * coeff_width)
            target_y_min = int(frame_y_min * coeff_height)
            target_y_max = int(frame_y_max * coeff_height)
            x_array = range(target_x_min, target_x_max, 1) #width or col
            y_array = range(target_y_min, target_y_max, 1) #height or row
            X, Y = np.meshgrid(x_array, y_array)
        

            Z = depth[Y,X]
            Z = np.where(Z > 600, Z, 700)  # Replace outliers with median

            #Save Z Z values as a heatmap for debugging
            plt.imshow(Z, cmap='viridis')
            plt.colorbar(label='Depth (mm)')
            plt.title(f'Depth Heatmap')
            plt.xlabel('X-axis (pixels)')
            plt.ylabel('Y-axis (pixels)')
            plt.savefig(f'depth_heatmap.png')
            plt.clf()  # Clear the figure for the next heatmap
        if z1 != 0 and z2 != 0:
            message.append(
                {
                    "object_number": index,
                    "x_1": x1,
                    "y_1": y1,
                    "z_1": z1,
                    "x_2": x2,
                    "y_2": y2,
                    "z_2": z2,
                }
            )

    return message
