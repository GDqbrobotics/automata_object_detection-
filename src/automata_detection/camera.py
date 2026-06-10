import time
from multiprocessing import Queue
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pyrealsense2 as rs
from PIL import Image

from .config import DEFAULT_CAMERA_CONFIG


class TemporalFilter:
    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = alpha
        self.previous_frame = None

    def process(self, frame: np.ndarray) -> np.ndarray:
        if self.previous_frame is None:
            result = frame
        else:
            result = cv2.addWeighted(frame, self.alpha, self.previous_frame, 1 - self.alpha, 0)
        self.previous_frame = result
        return result


class ImageCropper:
    def __init__(self, width: int, height: int, starting_row: int, starting_col: int) -> None:
        self.width = width
        self.height = height
        self.starting_row = starting_row
        self.starting_col = starting_col

    def crop(self, image: Image.Image) -> Image.Image:
        end_row = self.starting_row + self.height
        end_col = self.starting_col + self.width
        return image.crop((self.starting_col, self.starting_row, end_col, end_row))

    def cropped2orig(self, row: int, col: int) -> Tuple[int, int]:
        return row + self.starting_row, col + self.starting_col


def convert_depth_to_phys_coord_using_realsense(x: float, y: float, depth: float) -> Tuple[float, float, float]:
    intrinsics = rs.intrinsics()
    intrinsics.width = DEFAULT_CAMERA_CONFIG.width
    intrinsics.height = DEFAULT_CAMERA_CONFIG.height
    intrinsics.ppx = DEFAULT_CAMERA_CONFIG.K[2]
    intrinsics.ppy = DEFAULT_CAMERA_CONFIG.K[5]
    intrinsics.fx = DEFAULT_CAMERA_CONFIG.K[0]
    intrinsics.fy = DEFAULT_CAMERA_CONFIG.K[4]
    intrinsics.model = rs.distortion.none
    intrinsics.coeffs = DEFAULT_CAMERA_CONFIG.D
    result = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)
    return result[2], -result[0], -result[1]


def read_camera(*, frame_queue: Queue, width: int, height: int, verbose: bool = False) -> None:
    pipeline = rs.pipeline()
    config = rs.config()

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()

    found_rgb = any(
        s.get_info(rs.camera_info.name) == "RGB Camera" for s in device.sensors
    )
    if not found_rgb:
        raise RuntimeError("Depth camera with Color sensor required")

    config.enable_stream(rs.stream.depth, DEFAULT_CAMERA_CONFIG.width, DEFAULT_CAMERA_CONFIG.height, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 15)

    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    if verbose:
        print("[CAMERA] Depth Scale is:", depth_scale)

    saturation_sensor = profile.get_device().query_sensors()[1]
    saturation_sensor.set_option(rs.option.saturation, 70)
    saturation_sensor.set_option(rs.option.contrast, 65)
    saturation_sensor.set_option(rs.option.exposure, 45)

    align_to = rs.stream.color
    align = rs.align(align_to)
    temporal_filter = TemporalFilter(alpha=0.5)
    stale_frame_count = 0
    last_color_image = np.array([])

    while True:
        try:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
        except RuntimeError:
            print("[CAMERA] Detected stale frame condition - resetting camera")
            pipeline.stop()
            time.sleep(1)
            pipeline.start(config)
            continue

        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        if np.array_equal(last_color_image, color_image):
            stale_frame_count += 1
            if stale_frame_count > 5:
                print("[CAMERA] Detected stale frame condition - resetting camera")
                pipeline.stop()
                time.sleep(1)
                pipeline.start(config)
                stale_frame_count = 0
            continue

        stale_frame_count = 0
        last_color_image = color_image
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        depth_image = temporal_filter.process(depth_image)

        if verbose:
            print("[CAMERA] Read rgb frame of size", color_image.shape)
            print("[CAMERA] Read depth frame of size", depth_image.shape)

        if not frame_queue.full():
            frame_queue.put((color_image, depth_image))
