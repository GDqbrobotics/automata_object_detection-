import os
import time
import threading
from multiprocessing import Queue
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from .camera import DEFAULT_CAMERA_CONFIG, ImageCropper
from .model import extract_object, load_model
from .mqtt import MQTTClient
from .pose import find_min_segment, pixel2pose


CAPTURES_DIR = "captures"


class InferenceState:
    INITIALIZING = "initializing"
    READY = "ready"
    CAPTURING = "capturing"
    DETECTING = "detecting"  # New state: detecting objects for general capture


class CaptureSession:
    """Manages a capture session - collects frames and computes averaged poses."""
    
    def __init__(self, target_object_id: Optional[int] = None, num_frames: int = 10, verbose: bool = False):
        self.target_object_id = target_object_id
        self.num_frames = num_frames
        self.verbose = verbose
        self.frames_collected = 0
        self.accumulated_poses: Dict[int, List[Dict[str, float]]] = {}
        # Store frame data for each object to save cropped images later
        # obj_id -> List of (frame, contour, segment_p1, segment_p2, cx, cy)
        self.object_frame_data: Dict[int, List[Tuple[np.ndarray, np.ndarray, Tuple[int, int], Tuple[int, int], int, int]]] = {}
        
    def add_frame_results(self, results: List[Dict[str, float]], 
                          frame: np.ndarray, 
                          contours_data: List[Tuple[np.ndarray, Tuple[int, int], Tuple[int, int], int, int]]) -> bool:
        """Add results from one frame. Returns True if capture is complete."""
        self.frames_collected += 1
        
        for pose, contour_data in zip(results, contours_data):
            obj_id = pose.get("object_number", 0)
            # If target_object_id is specified, only collect for that object
            if self.target_object_id is not None and obj_id != self.target_object_id:
                continue
                
            if obj_id not in self.accumulated_poses:
                self.accumulated_poses[obj_id] = []
            self.accumulated_poses[obj_id].append(pose)
            
            # Store frame data for cropping later
            if obj_id not in self.object_frame_data:
                self.object_frame_data[obj_id] = []
            self.object_frame_data[obj_id].append((frame.copy(), *contour_data))
            
        if self.verbose:
            print(f"[CAPTURE] Collected frame {self.frames_collected}/{self.num_frames}, "
                  f"objects: {list(self.accumulated_poses.keys())}")
        
        return self.frames_collected >= self.num_frames
    
    def get_averaged_results(self) -> List[Dict[str, float]]:
        """Compute averaged poses from all collected frames."""
        averaged = []
        for obj_id, poses in self.accumulated_poses.items():
            if not poses:
                continue
                
            # Average all coordinate fields
            avg_pose = {"object_number": obj_id}
            for key in ["x_1", "y_1", "z_1", "x_2", "y_2", "z_2"]:
                values = [p[key] for p in poses if key in p and p[key] != 0]
                if values:
                    avg_pose[key] = sum(values) / len(values)
                else:
                    avg_pose[key] = 0.0
            averaged.append(avg_pose)
            
        if self.verbose:
            print(f"[CAPTURE] Averaged results: {averaged}")
        return averaged
    
    def save_cropped_images(self, base_frame: np.ndarray) -> List[str]:
        """Save cropped images for each captured object.
        
        Args:
            base_frame: The original full frame (before cropping) for coordinate mapping
            
        Returns:
            List of saved file paths
        """
        os.makedirs(CAPTURES_DIR, exist_ok=True)
        saved_paths = []
        
        for obj_id, frame_data_list in self.object_frame_data.items():
            if not frame_data_list:
                continue
                
            # Use the middle frame for the best representation
            mid_idx = len(frame_data_list) // 2
            frame, contour, segment_p1, segment_p2, cx, cy = frame_data_list[mid_idx]
            
            # Compute bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)
            
            # Add some padding around the object
            padding = 10
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)
            
            # Crop the object region
            cropped = frame[y1:y2, x1:x2]
            
            if cropped.size == 0:
                if self.verbose:
                    print(f"[CAPTURE] Warning: Empty crop for object {obj_id}")
                continue
            
            # Save as JPG
            timestamp = int(time.time() * 1000)
            filename = f"object_{obj_id}_{timestamp}.jpg"
            filepath = os.path.join(CAPTURES_DIR, filename)
            
            cv2.imwrite(filepath, cropped)
            saved_paths.append(filepath)
            
            if self.verbose:
                print(f"[CAPTURE] Saved cropped image for object {obj_id}: {filepath} ({cropped.shape})")
        
        return saved_paths
    
    def is_complete(self) -> bool:
        return self.frames_collected >= self.num_frames


def run_detection(frame: np.ndarray, cropper: ImageCropper, model) -> Tuple[List[int], List[Tuple[np.ndarray, Tuple[int, int], Tuple[int, int], int, int]], np.ndarray]:
    """Run object detection on a frame and return detected object IDs, contours data, and annotated image.
    
    Returns:
        Tuple of (object_ids, contours_data, cv_image)
        object_ids: List of sequential object IDs (0, 1, 2, ...)
        contours_data: List of (contour, segment_p1, segment_p2, cx, cy)
        cv_image: Annotated image for debug
    """
    base_image = Image.fromarray(frame)
    base_image = cropper.crop(base_image)
    
    annotated_image, _ = extract_object(model, base_image)
    cv_image = np.array(annotated_image.convert("RGBA"))
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGBA2BGRA)
    
    _, thresholded = cv2.threshold(cv_image[:, :, 3], 254, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    object_ids = []
    contours_data = []
    for contour in contours:
        if contour.size < 180 or contour.size > 1500:
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
        
        object_ids.append(len(object_ids))  # Sequential IDs based on detection order
        contours_data.append((contour, segment_p1, segment_p2, cx, cy))
    
    return object_ids, contours_data, cv_image


def start_inference(*, frame_queue: Queue, parameters_queue: Queue, send_queue: Queue = None, 
                    verbose: bool = False, sleep: float = 0.0, depth_height: int = 720, 
                    depth_width: int = 1280, camera_type: str = "realsense",
                    mqtt_host: str = "192.168.139.70", mqtt_port: int = 1883,
                    mqtt_user: str = "mqtt", mqtt_password: str = "Vn370gi@lo#T",
                    mqtt_status_topic: str = "/status", mqtt_results_topic: str = "/results",
                    mqtt_command_topic: str = "/command", capture_frames: int = 10) -> None:
    """
    Main inference loop with MQTT state machine.
    
    States:
    - INITIALIZING: Starting up, loading model, connecting to camera
    - READY: Model loaded, camera running, waiting for capture command
    - DETECTING: Detecting objects for general capture (no object_id specified)
    - CAPTURING: Collecting N frames, computing averaged poses, saving cropped images
    
    MQTT Topics:
    - /status: Publishes {"status": "INITIALIZING|READY|DETECTING|CAPTURING"}
    - /results: Publishes object_points messages (one per object) and objects_detected message
    - /command: Subscribes to {"event": "capture", "object_id": optional}
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device=device)

    cropper = ImageCropper(
        DEFAULT_CAMERA_CONFIG.crop_width,
        DEFAULT_CAMERA_CONFIG.crop_height,
        DEFAULT_CAMERA_CONFIG.crop_starting_row,
        DEFAULT_CAMERA_CONFIG.crop_starting_col,
    )

    depth_intrinsics = None
    extrinsic = None
    
    # MQTT Client for status/results publishing and command subscription
    mqtt_client = MQTTClient(
        host=mqtt_host,
        port=mqtt_port,
        status_topic=mqtt_status_topic,
        results_topic=mqtt_results_topic,
        command_topic=mqtt_command_topic,
        username=mqtt_user,
        password=mqtt_password,
        verbose=verbose,
    )
    
    # State management
    current_state = InferenceState.INITIALIZING
    capture_session: Optional[CaptureSession] = None
    state_lock = threading.Lock()
    latest_command: Optional[Dict] = None
    command_lock = threading.Lock()
    
    # State for handling general capture (detect objects first)
    pending_general_capture = False
    detected_object_ids: List[int] = []
    detected_contours_data: List[Tuple[np.ndarray, Tuple[int, int], Tuple[int, int], int, int]] = []
    
    def handle_command(payload: Dict):
        """Callback for incoming MQTT commands."""
        nonlocal latest_command
        with command_lock:
            latest_command = payload
        if verbose:
            print(f"[INFERENCE] Received command: {payload}")
    
    mqtt_client.set_command_callback(handle_command)
    mqtt_client.connect()
    
    # Publish initial status
    mqtt_client.publish_status(InferenceState.INITIALIZING)
    
    # Wait for camera parameters (especially for Orbbec)
    print("[INFERENCE] Waiting for camera parameters...")
    param_timeout = time.time() + 10
    while camera_type == "orbbec" and (depth_intrinsics is None or extrinsic is None):
        if not parameters_queue.empty():
            from .camera_orbbec import OBCameraIntrinsic, OBExtrinsic
            parameters = parameters_queue.get()
            depth_intrinsics = OBCameraIntrinsic()
            extrinsic = OBExtrinsic()
            depth_intrinsics.fx = parameters.fx
            depth_intrinsics.fy = parameters.fy
            depth_intrinsics.cx = parameters.cx
            depth_intrinsics.cy = parameters.cy
            depth_intrinsics.width = parameters.width
            depth_intrinsics.height = parameters.height
            extrinsic.rot = parameters.rot
            extrinsic.transform = parameters.transform
            color_width = parameters.color_width
            color_height = parameters.color_height
            print("[INFERENCE] Camera parameters received")
            break
        if time.time() > param_timeout:
            print("[INFERENCE] Warning: Camera parameters not received within timeout")
            break
        time.sleep(0.1)
    
    # Transition to READY state
    with state_lock:
        current_state = InferenceState.READY
    mqtt_client.publish_status(InferenceState.READY)
    print("[INFERENCE] System READY - waiting for capture commands")
    
    try:
        while True:
            # Process incoming commands
            with command_lock:
                command = latest_command
                latest_command = None
            
            if command:
                event = command.get("event", "").lower()
                if "capture" in event:
                    object_id = command.get("object_id")
                    with state_lock:
                        if current_state == InferenceState.READY:
                            if object_id is None:
                                # General capture: need to detect objects first
                                pending_general_capture = True
                                print(f"[INFERENCE] General capture requested - will detect objects first")
                            else:
                                # Specific object capture: start directly
                                current_state = InferenceState.CAPTURING
                                capture_session = CaptureSession(
                                    target_object_id=object_id,
                                    num_frames=capture_frames,
                                    verbose=verbose
                                )
                                mqtt_client.publish_status(InferenceState.CAPTURING)
                                print(f"[INFERENCE] Starting CAPTURING phase for object_id={object_id} ({capture_frames} frames)")
                        else:
                            print(f"[INFERENCE] Ignoring capture command - current state: {current_state}")
            
            # Get frame from camera
            if frame_queue.empty():
                time.sleep(sleep)
                continue
            
            frame, depth = frame_queue.get()
            # Keep a copy of the original full frame for cropping
            original_frame = frame.copy()
            coeff_height = depth_height / frame.shape[0]
            coeff_width = depth_width / frame.shape[1]
            
            # Handle DETECTING state - run detection to find all objects
            with state_lock:
                state = current_state
            
            if state == InferenceState.READY and pending_general_capture:
                # Switch to DETECTING state and run detection on this frame
                with state_lock:
                    current_state = InferenceState.DETECTING
                mqtt_client.publish_status(InferenceState.DETECTING)
                print("[INFERENCE] DETECTING objects...")
                
                # Run detection
                detected_object_ids, detected_contours_data, cv_image = run_detection(frame, cropper, model)
                
                if detected_object_ids:
                    # Publish objects_detected message
                    mqtt_client.publish_objects_detected(detected_object_ids)
                    print(f"[INFERENCE] Detected {len(detected_object_ids)} objects: {detected_object_ids}")
                    
                    # Start CAPTURING for all detected objects
                    with state_lock:
                        current_state = InferenceState.CAPTURING
                        capture_session = CaptureSession(
                            target_object_id=None,  # Capture all detected objects
                            num_frames=capture_frames,
                            verbose=verbose
                        )
                    mqtt_client.publish_status(InferenceState.CAPTURING)
                    print(f"[INFERENCE] Starting CAPTURING phase for all {len(detected_object_ids)} objects ({capture_frames} frames)")
                    pending_general_capture = False
                else:
                    print("[INFERENCE] No objects detected - returning to READY")
                    with state_lock:
                        current_state = InferenceState.READY
                    mqtt_client.publish_status(InferenceState.READY)
                    pending_general_capture = False
                
                # Save debug image and continue to next frame
                cv2.imwrite("result.png", cv_image)
                if sleep > 0:
                    time.sleep(sleep)
                continue
            
            # Normal frame processing for CAPTURING or READY state
            base_image = Image.fromarray(frame)
            base_image = cropper.crop(base_image)
            
            if verbose:
                print("[INFERENCE] Inference on image of size", base_image.size)
            
            # Run object detection
            annotated_image, _ = extract_object(model, base_image)
            cv_image = np.array(annotated_image.convert("RGBA"))
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGBA2BGRA)
            
            _, thresholded = cv2.threshold(cv_image[:, :, 3], 254, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            
            results = []
            contours_data = []  # Store contour data for cropping later
            for contour in contours:
                if contour.size < 180 or contour.size > 1500:
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
                # Store data needed for cropping: contour, segment points, centroid
                contours_data.append((contour, segment_p1, segment_p2, cx, cy))
            
            # Convert pixel coordinates to 3D poses
            filtered_depth = cv2.medianBlur(depth, 5)
            filtered_depth = cv2.medianBlur(filtered_depth, 5)
            message = pixel2pose(
                results, filtered_depth, coeff_height, coeff_width, 
                camera_type=camera_type, depth_intrinsics=depth_intrinsics, extrinsic=extrinsic
            )
            
            # Handle capturing state
            with state_lock:
                state = current_state
                session = capture_session
            
            if state == InferenceState.CAPTURING and session and message:
                # Pass both message and contour data for frame storage
                capture_complete = session.add_frame_results(message, cv_image, contours_data)
                
                if capture_complete:
                    # Compute averaged results and publish
                    averaged_results = session.get_averaged_results()
                    mqtt_client.publish_results(averaged_results)
                    
                    # Save cropped images for each captured object
                    saved_images = session.save_cropped_images(original_frame)
                    if saved_images and verbose:
                        print(f"[INFERENCE] Saved {len(saved_images)} cropped object images to {CAPTURES_DIR}/")
                    
                    # Return to READY state
                    with state_lock:
                        current_state = InferenceState.READY
                        capture_session = None
                    mqtt_client.publish_status(InferenceState.READY)
                    print("[INFERENCE] Capture complete - returned to READY state")
            
            # Save debug image
            cv2.imwrite("result.png", cv_image)
            
            if sleep > 0:
                time.sleep(sleep)
                
    except KeyboardInterrupt:
        print("[INFERENCE] Interrupted")
    except Exception as e:
        print(f"[INFERENCE] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        mqtt_client.disconnect()
        print("[INFERENCE] Shutdown complete")