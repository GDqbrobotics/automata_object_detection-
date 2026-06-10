import time
from multiprocessing import Queue
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from .camera import DEFAULT_CAMERA_CONFIG, ImageCropper
from .model import extract_object, load_model
from .pose import find_min_segment, pixel2pose


def start_inference(*, frame_queue: Queue, send_queue: Queue, verbose: bool = False, sleep: float = 0.0, depth_height: int = 720, depth_width: int = 1280, camera_type: str = "realsense") -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device=device)

    cropper = ImageCropper(
        DEFAULT_CAMERA_CONFIG.crop_width,
        DEFAULT_CAMERA_CONFIG.crop_height,
        DEFAULT_CAMERA_CONFIG.crop_starting_row,
        DEFAULT_CAMERA_CONFIG.crop_starting_col,
    )

    while True:
        if frame_queue.empty():
            time.sleep(sleep)
            continue

        frame, depth = frame_queue.get()
        coeff_height = depth_height / frame.shape[0]
        coeff_width = depth_width / frame.shape[1]

        base_image = Image.fromarray(frame)
        base_image = cropper.crop(base_image)

        if verbose:
            print("[INFERENCE] Inference on image of size", base_image.size)

        annotated_image, _ = extract_object(model, base_image)
        cv_image = np.array(annotated_image.convert("RGBA"))
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGBA2BGRA)

        _, thresholded = cv2.threshold(cv_image[:, :, 3], 254, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        results = []
        #canvas = np.ones_like(cv_image, dtype=np.uint8) * 255 #white background

        for contour in contours:
            if contour.size < 300 or contour.size > 1500:
                continue

            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.drawContours(cv_image, [contour], -1, (255, 0, 0, 255), 2)
            cv2.circle(cv_image, (cx, cy), 6, (0, 255, 0, 255), 2)

            segment_p1, segment_p2 = find_min_segment(cx, cy, contour)
            cv2.line(cv_image, segment_p1, segment_p2, (0, 255, 0), 2)
            cv2.circle(cv_image, segment_p1, 7, (0, 255, 255), -1)
            cv2.circle(cv_image, segment_p2, 7, (0, 255, 255), -1)

            results.append({"1": segment_p1, "2": segment_p2})

        message = pixel2pose(results, depth, coeff_height, coeff_width, camera_type=camera_type)
        if message:
            send_queue.put(message)

        timestamp = int(time.time())
        cv2.putText(cv_image, str(timestamp), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite("result.png", cv_image)

        if sleep > 0:
            time.sleep(sleep)
