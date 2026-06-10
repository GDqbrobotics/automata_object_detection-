import time
from multiprocessing import Queue
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
from typing import Union, Any, Optional

from .config import DEFAULT_CAMERA_CONFIG

try:
    from pyorbbecsdk import *
except ImportError:
    raise ImportError("pyorbbecsdk is required for Orbbec camera support. Install it with: pip install pyorbbecsdk")

_initialized = False
MIN_DEPTH = 20  # 20mm
MAX_DEPTH = 10000  # 10000mm

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


def convert_depth_to_phys_coord_using_orbbec(x: float, y: float, depth: float) -> Tuple[float, float, float]:
    """Convert depth pixel coordinates to physical 3D coordinates using Orbbec camera intrinsics."""
    fx = DEFAULT_CAMERA_CONFIG.K[0]
    fy = DEFAULT_CAMERA_CONFIG.K[4]
    ppx = DEFAULT_CAMERA_CONFIG.K[2]
    ppy = DEFAULT_CAMERA_CONFIG.K[5]

    z = depth
    x_phys = (x - ppx) * z / fx
    y_phys = (y - ppy) * z / fy

    return z, -x_phys, -y_phys


def frame_to_bgr_image(frame: VideoFrame) -> Union[Optional[np.array], Any]:
    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()
    data = np.asanyarray(frame.get_data())
    image = np.zeros((height, width, 3), dtype=np.uint8)
    if color_format == OBFormat.RGB:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif color_format == OBFormat.BGR:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_format == OBFormat.YUYV:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_YUYV)
    elif color_format == OBFormat.MJPG:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif color_format == OBFormat.I420:
        image = i420_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.NV12:
        image = nv12_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.NV21:
        image = nv21_to_bgr(data, width, height)
        return image
    elif color_format == OBFormat.UYVY:
        image = np.resize(data, (height, width, 2))
        image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR_UYVY)
    else:
        print("Unsupported color format: {}".format(color_format))
        return None
    return image

def get_frame_data(color_frame, depth_frame):

    global _initialized

    color_frame = color_frame.as_video_frame()
    depth_frame = depth_frame.as_video_frame()

    depth_width = depth_frame.get_width()
    depth_height = depth_frame.get_height()

    color_width = color_frame.get_width()
    color_height = color_frame.get_height()

    color_profile = color_frame.get_stream_profile()
    depth_profile = depth_frame.get_stream_profile()
    print("video profile:", color_profile.as_video_stream_profile())
    color_intrinsics = color_profile.as_video_stream_profile().get_intrinsic()
    color_distortion = color_profile.as_video_stream_profile().get_distortion()
    depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsic()
    depth_distortion = depth_profile.as_video_stream_profile().get_distortion()

    print("depth intrinsics:", depth_intrinsics)

    extrinsic = depth_profile.get_extrinsic_to(color_profile)

    print("extrinsic:", extrinsic)
    _initialized = True
    return color_width, color_height, depth_width, depth_height, color_intrinsics, color_distortion, depth_intrinsics, depth_distortion, extrinsic

def read_camera(*, frame_queue,  width, height, verbose=False):
    # Create a pipeline with default device
    pipeline = Pipeline()
    temporal_filter = TemporalFilter(alpha=0.5)
    config = Config()  # Initialize the config for the pipeline
    align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
    try:
        # Enable depth and color sensors
        for sensor_type in [OBSensorType.DEPTH_SENSOR, OBSensorType.COLOR_SENSOR]:
            profile_list = pipeline.get_stream_profile_list(sensor_type)
            assert profile_list is not None
            profile = profile_list.get_default_video_stream_profile()
            try:
                for profile_iterator in profile_list:
                    if profile_iterator.get_width() == width and profile_iterator.get_height() == height:
                        profile = profile_iterator
                        break
            except Exception as e:
                print(e)
            assert profile is not None
            print(f"{sensor_type} profile:", profile)
            config.enable_stream(profile)  # Enable the stream for the sensor
    except Exception as e:
        print(e)
        return

    print("start pipeline")
    pipeline.start(config)  # Start the pipeline with the config

    while True:
        # Wait for frames from the pipeline (with a timeout of 100 ms)
        frames = pipeline.wait_for_frames(100)
        if not frames:
            continue

        # --- Spatial Alignment ---
        # Transforms one stream to the coordinate system/FOV of the other
        frames = align_filter.process(frames)
        if not frames:
            continue
        
        frames = frames.as_frame_set()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            continue
        if verbose: print("[STREAM] Read rgb frame of size", color_frame.get_width(), color_frame.get_height())
        if verbose: print("[STREAM] Read depth frame of size", depth_frame.get_width(), depth_frame.get_height())

        if not _initialized: 
            _color_width, _color_height, _depth_width, _depth_height, _color_intrinsics, _color_distortion, _depth_intrinsics, _depth_distortion, _extrinsic = get_frame_data(color_frame, depth_frame)
                
        # the depth frame has lower resolution than the color frame, so we need to resize it
        # to match the size of the color frame. We use the nearest neighbor interpolation
        # to avoid creating new data points (which could lead to incorrect depth values)
        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(_depth_height, _depth_width)
        
        depth_data = depth_data.astype(np.float32) * depth_frame.get_depth_scale()
        depth_data = np.where((depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH), depth_data, 0)
        depth_data = depth_data.astype(np.uint16)

        # Apply temporal filtering
        depth = temporal_filter.process(depth_data)
        
        image = frame_to_bgr_image(color_frame)

        if not frame_queue.full():
            frame_queue.put((image, depth))

