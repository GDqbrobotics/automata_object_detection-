import time
from multiprocessing import Queue
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

from .camera import DEFAULT_CAMERA_CONFIG
from .config import (
    ARTIFACT_COLORS,
    DEFAULT_DEPTH_FILTER_CONFIG,
    DEFAULT_DETECTION_CONFIG,
    DEFAULT_TRACKER_CONFIG,
)
from .model import extract_object, load_model
from .pose import find_min_segment, segment_to_pose
from .tracker import ArtifactTracker


# Width of the ring around a blob used as its local table baseline (full-frame px),
# and how many valid depth pixels are needed before a blob's height can be judged:
# at least MIN_HEIGHT_PIXELS, and at least MIN_VALID_FRACTION of the blob's area.
# The fraction matters for dark material with no IR return: its few valid pixels
# are rim pixels at mat level, which would wrongly measure the blob as flat.
LOCAL_RING_PX = 20
MIN_HEIGHT_PIXELS = 50
MIN_VALID_FRACTION = 0.3


def apply_matrix(matrix, x, y):
    """Map one crop pixel to full-frame pixels through the crop->frame matrix."""
    px = matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]
    py = matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]
    return int(px), int(py)


def crop_region_of_depth(depth: np.ndarray, crop_matrix, crop_width: int, crop_height: int, coeff_height: float, coeff_width: float) -> np.ndarray:
    """Return the part of the depth image that corresponds to the current crop."""
    corners = [
        apply_matrix(crop_matrix, 0, 0),
        apply_matrix(crop_matrix, crop_width, 0),
        apply_matrix(crop_matrix, crop_width, crop_height),
        apply_matrix(crop_matrix, 0, crop_height),
    ]
    xs = [corner[0] for corner in corners]
    ys = [corner[1] for corner in corners]
    row0 = max(int(min(ys) * coeff_height), 0)
    row1 = int(max(ys) * coeff_height)
    col0 = max(int(min(xs) * coeff_width), 0)
    col1 = int(max(xs) * coeff_width)
    return depth[row0:row1, col0:col1]


def update_table_depth(depth_region: np.ndarray, current_estimate: Optional[float], disagree_count: int):
    """Keep an estimate of the table depth (the flat work surface).

    Uses the median of the valid depth pixels in the crop: the fragments cover
    only a small part of the surface, so the median lands on the table itself.
    The estimate moves slowly (EMA) and only when the scene looks undisturbed —
    a big jump of the median usually means a hand/arm is in the way. But if the
    median keeps disagreeing for many cycles in a row (the crop moved to an area
    of the table at a different depth), the estimate is re-learned from scratch
    instead of staying stuck forever.

    Returns (estimate, disagree_count).
    """
    values = depth_region[depth_region > 0]
    if values.size == 0:
        return current_estimate, disagree_count
    median = float(np.median(values))
    if current_estimate is None:
        return median, 0
    if abs(median - current_estimate) < 15.0:
        alpha = DEFAULT_DEPTH_FILTER_CONFIG.table_alpha
        return (1.0 - alpha) * current_estimate + alpha * median, 0
    disagree_count += 1
    if disagree_count >= DEFAULT_DEPTH_FILTER_CONFIG.table_relearn_cycles:
        return median, 0
    return current_estimate, disagree_count


def blob_height_over_table(contour_full: np.ndarray, depth: np.ndarray, coeff_height: float, coeff_width: float, table_depth: float) -> float:
    """Return how much a blob sticks out of the table, in depth units (mm).

    We look at the depth of the pixels inside the blob outline (full-frame
    coordinates) and take a low percentile (a point close to the camera): a flat
    fragment gives a small height, a hand or a robot arm gives a large one even
    when its blob is fused with a fragment's blob.
    """
    x, y, w, h = cv2.boundingRect(contour_full)
    if w <= 0 or h <= 0:
        return 0.0
    mask = np.zeros((h, w), dtype=np.uint8)
    shifted = (contour_full - [x, y]).astype(np.int32)
    cv2.drawContours(mask, [shifted], -1, 255, -1)

    ys, xs = np.nonzero(mask)
    # Full-frame pixels -> depth image pixels.
    depth_x = np.clip(((xs + x) * coeff_width).astype(int), 0, depth.shape[1] - 1)
    depth_y = np.clip(((ys + y) * coeff_height).astype(int), 0, depth.shape[0] - 1)

    values = depth[depth_y, depth_x]
    values = values[values > 0]
    if values.size == 0:
        # No valid depth in the blob: we cannot judge it, treat it as flat.
        return 0.0
    top = float(np.percentile(values, 20))
    return table_depth - top


def blob_height_local(contour_full: np.ndarray, depth: np.ndarray, coeff_height: float, coeff_width: float) -> Optional[float]:
    """Return how much a blob rises above its LOCAL surroundings, in mm.

    Compares the median depth inside the blob with the median depth of a ring
    around it. Using the local ring as the baseline (instead of the global table
    plane) cancels the table's tilt and most of the depth noise, so even a thin
    fragment measures a small positive height, while a phantom "blob" made of
    mat texture measures about zero. Returns None when there are not enough
    valid depth pixels to judge (e.g. dark material with no IR return) - in that
    case the caller must keep the blob, never reject it.
    """
    x, y, w, h = cv2.boundingRect(contour_full)
    ring = LOCAL_RING_PX
    x0 = x - ring
    y0 = y - ring
    box_w = w + 2 * ring
    box_h = h + 2 * ring

    # Mask of the blob inside the expanded box: 255 = blob, 0 = surroundings.
    mask = np.zeros((box_h, box_w), dtype=np.uint8)
    shifted = (contour_full - [x0, y0]).astype(np.int32)
    cv2.drawContours(mask, [shifted], -1, 255, -1)

    def valid_depth_values(xs, ys):
        depth_x = np.clip(((xs + x0) * coeff_width).astype(int), 0, depth.shape[1] - 1)
        depth_y = np.clip(((ys + y0) * coeff_height).astype(int), 0, depth.shape[0] - 1)
        values = depth[depth_y, depth_x]
        return values[values > 0]

    inside_ys, inside_xs = np.nonzero(mask)
    around_ys, around_xs = np.nonzero(mask == 0)
    inside = valid_depth_values(inside_xs, inside_ys)
    around = valid_depth_values(around_xs, around_ys)
    if inside.size < MIN_HEIGHT_PIXELS or around.size < MIN_HEIGHT_PIXELS:
        return None
    if inside.size < MIN_VALID_FRACTION * inside_xs.size:
        return None
    return float(np.median(around)) - float(np.median(inside))


def start_inference(*, frame_queue: Queue, parameters_queue: Queue, send_queue: Queue, objects_queue: Queue = None, crop_queue: Queue = None, result_frame_queue: Queue = None, verbose: bool = False, sleep: float = 0.0, depth_height: int = 720, depth_width: int = 1280, camera_type: str = "realsense") -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device=device)

    # The crop is described by an affine matrix that maps crop pixels to
    # full-frame pixels. At startup it is the static crop from the config (a
    # plain translation); as soon as the ArUco node finds the grid it sends a
    # rectified crop (the straightened grid + margin) and the crop follows the
    # grid from then on, whatever its rotation.
    crop_matrix = np.array([
        [1.0, 0.0, float(DEFAULT_CAMERA_CONFIG.crop_starting_col)],
        [0.0, 1.0, float(DEFAULT_CAMERA_CONFIG.crop_starting_row)],
    ])
    crop_width = DEFAULT_CAMERA_CONFIG.crop_width
    crop_height = DEFAULT_CAMERA_CONFIG.crop_height
    crop_margin = 0  # the static fallback crop has no margin ring around the grid

    # Tracks each artifact across cycles so a placed artifact keeps its pose and
    # color instead of being re-estimated (and re-colored) every BiRefNet cycle.
    tracker = ArtifactTracker(
        match_distance_px=DEFAULT_TRACKER_CONFIG.match_distance_px,
        miss_limit=DEFAULT_TRACKER_CONFIG.miss_limit,
        reacquire_window_s=DEFAULT_TRACKER_CONFIG.reacquire_window_s,
        confirm_hits=DEFAULT_TRACKER_CONFIG.confirm_hits,
        min_height_mm=DEFAULT_DEPTH_FILTER_CONFIG.min_artifact_height_mm,
    )
    last_publish_time = time.time()

    depth_intrinsics = None
    extrinsic = None

    table_depth = None        # estimated depth of the work surface (mm)
    table_disagree_count = 0  # cycles in a row the measurement disagreed with the estimate
    scene_occluded = False    # True while a hand/arm is over the table

    while True:
        if frame_queue.empty():
            time.sleep(sleep)
            continue

        if camera_type == "orbbec" and not parameters_queue.empty():
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

        # Follow the grid: the ArUco node sends a rectified crop (matrix + size)
        # whenever the grid really moves.
        if crop_queue is not None and not crop_queue.empty():
            crop = crop_queue.get()
            crop_matrix = crop["matrix"]
            crop_width = crop["width"]
            crop_height = crop["height"]
            crop_margin = crop["margin"]
            if verbose:
                print("[INFERENCE] Crop follows the grid: %dx%d px" % (crop_width, crop_height))

        frame, depth = frame_queue.get()
        coeff_height = depth_height / frame.shape[0]
        coeff_width = depth_width / frame.shape[1]

        # Filter the depth once and use it for everything this cycle
        # (table estimate, hand filter, pose estimation).
        filtered_depth = cv2.medianBlur(depth, 5)  # Kernel size of 5

        depth_region = crop_region_of_depth(filtered_depth, crop_matrix, crop_width, crop_height, coeff_height, coeff_width)
        table_depth, table_disagree_count = update_table_depth(depth_region, table_depth, table_disagree_count)

        # Rectified crop: warp the frame so the grid fills the crop, straight,
        # whatever its rotation in the camera image. BiRefNet only ever sees the
        # mat (plus the small margin), like it did with a well-tuned static crop.
        warped = cv2.warpAffine(frame, crop_matrix, (crop_width, crop_height), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        base_image = Image.fromarray(warped)

        if verbose:
            print("[INFERENCE] Inference on image of size", base_image.size)

        annotated_image, _ = extract_object(model, base_image)
        cv_image = np.array(annotated_image.convert("RGBA"))
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGBA2BGRA)

        _, thresholded = cv2.threshold(cv_image[:, :, 3], DEFAULT_DETECTION_CONFIG.alpha_threshold, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # Build one detection per valid contour: centroid and grasp segment in
        # full-frame pixels (used for matching and pose, so they stay valid even
        # when the crop moves) plus crop-local copies to draw on result.png.
        # A blob that sticks out of the table too much is a hand or a robot arm,
        # not a fragment: it is skipped and it marks the scene as occluded.
        detections = []
        tall_blob_found = False
        for contour in contours:
            if contour.size < DEFAULT_DETECTION_CONFIG.min_contour_size or contour.size > DEFAULT_DETECTION_CONFIG.max_contour_size:
                continue

            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Blobs centered in the margin ring around the grid are not fragments
            # (fragments sit inside the grid): mat border, marker sheets, table.
            if crop_margin > 0:
                if cx < crop_margin or cy < crop_margin or cx > crop_width - crop_margin or cy > crop_height - crop_margin:
                    continue

            # Full-frame outline of this blob, mapped through the crop matrix.
            contour_full = cv2.transform(contour.astype(np.float32), crop_matrix).astype(np.int32)

            if table_depth is not None:
                height = blob_height_over_table(contour_full, filtered_depth, coeff_height, coeff_width, table_depth)
                if height > DEFAULT_DEPTH_FILTER_CONFIG.max_artifact_height_mm:
                    tall_blob_found = True
                    if verbose:
                        print("[INFERENCE] Skipping a blob %.0f mm above the table (hand/arm?)" % height)
                    continue

            segment_p1, segment_p2 = find_min_segment(cx, cy, contour)

            detections.append({
                # Full-frame coordinates (for the tracker and the pose).
                "centroid": apply_matrix(crop_matrix, cx, cy),
                "segment": {
                    "1": apply_matrix(crop_matrix, segment_p1[0], segment_p1[1]),
                    "2": apply_matrix(crop_matrix, segment_p2[0], segment_p2[1]),
                },
                "contour": contour_full,
                # Rise above the local surroundings (mm, or None if not judgeable):
                # the tracker averages it over the confirmation cycles and drops
                # flat "phantom" blobs that are just mat texture.
                "height_mm": blob_height_local(contour_full, filtered_depth, coeff_height, coeff_width),
                # Crop-local coordinates (only to draw on this cycle's result.png).
                "centroid_crop": (cx, cy),
                "segment_crop": {"1": segment_p1, "2": segment_p2},
                "contour_crop": contour,
            })

        # The scene counts as occluded when a tall blob was seen, or when a big
        # part of the crop is much closer to the camera than the table (an arm
        # that BiRefNet did not segment still shows up in the depth).
        occluded = tall_blob_found
        if table_depth is not None and depth_region.size > 0:
            valid = depth_region > 0
            tall = valid & (depth_region < table_depth - DEFAULT_DEPTH_FILTER_CONFIG.max_artifact_height_mm)
            valid_count = int(valid.sum())
            if valid_count > 0 and tall.sum() / valid_count > DEFAULT_DEPTH_FILTER_CONFIG.occlusion_freeze_fraction:
                occluded = True
        if verbose and occluded != scene_occluded:
            print("[INFERENCE] Scene occluded:", occluded)
        scene_occluded = occluded

        # Match detections to already tracked artifacts (or create new ones / drop
        # one missing for too long). Matched artifacts keep their pose untouched:
        # re-detecting the grid or re-running BiRefNet must never move an artifact
        # that has already been placed. While the scene is occluded the misses are
        # frozen, so a covered artifact never expires during a pick/place.
        before_ids = {track_id for track_id, track in tracker.tracks.items() if track.pose_valid}
        added_ids, removed_ids, detection_track_ids = tracker.update(detections, freeze_misses=occluded)
        if verbose and (added_ids or removed_ids):
            print("[INFERENCE] Tracks added:", added_ids, "removed:", removed_ids)

        # Estimate the pose once for any artifact that does not have a valid one yet
        # (a brand new artifact, or one whose depth reading was invalid last time).
        for track in tracker.tracks.values():
            if track.pose_valid:
                continue
            pose = segment_to_pose(
                track.segment, filtered_depth, coeff_height, coeff_width,
                camera_type=camera_type, depth_intrinsics=depth_intrinsics, extrinsic=extrinsic,
            )
            if pose is not None:
                track.pose = pose
                track.pose_valid = True

        # Draw each of this cycle's detections with its artifact's stable color and id.
        for detection, track_id in zip(detections, detection_track_ids):
            color = ARTIFACT_COLORS[track_id % len(ARTIFACT_COLORS)] if track_id is not None else (255, 0, 0)
            cx, cy = detection["centroid_crop"]
            segment_p1 = detection["segment_crop"]["1"]
            segment_p2 = detection["segment_crop"]["2"]
            cv2.drawContours(cv_image, [detection["contour_crop"]], -1, (*color, 255), 2)
            cv2.circle(cv_image, (cx, cy), 6, (*color, 255), 2)
            cv2.line(cv_image, segment_p1, segment_p2, (*color, 255), 2)
            cv2.circle(cv_image, segment_p1, 7, (0, 255, 255), -1)
            cv2.circle(cv_image, segment_p2, 7, (0, 255, 255), -1)
            if track_id is not None:
                cv2.putText(cv_image, str(track_id), (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (*color, 255), 2, cv2.LINE_AA)

        # Send every currently tracked artifact's outline and pose to the ArUco
        # node, built from the tracker (not just this cycle's fresh detections)
        # so a briefly occluded artifact keeps being sent with its last known
        # outline instead of disappearing from the grid while it is only
        # temporarily missed. The pose is shown on the web page's info panel.
        if objects_queue is not None and not objects_queue.full():
            objects_queue.put([
                {"id": track_id, "contour": track.contour, "pose": track.pose if track.pose_valid else None}
                for track_id, track in tracker.tracks.items()
            ])

        # Publish the full artifact list only when something actually changed (an
        # artifact appeared, disappeared, or got its pose for the first time), plus
        # a periodic heartbeat so a consumer that missed a message stays in sync.
        after_ids = {track_id for track_id, track in tracker.tracks.items() if track.pose_valid}
        now = time.time()
        changed = before_ids != after_ids
        heartbeat_due = now - last_publish_time >= DEFAULT_TRACKER_CONFIG.heartbeat_s
        if changed or heartbeat_due:
            message = [
                {"object_number": track_id, **tracker.tracks[track_id].pose}
                for track_id in sorted(after_ids)
            ]
            send_queue.put(message)
            last_publish_time = now

        timestamp = int(time.time())
        cv2.putText(cv_image, str(timestamp), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite("result.png", cv_image)

        # Also send the annotated frame to the ArUco node, which shows it as a
        # live video next to the grid view on the web page.
        if result_frame_queue is not None and not result_frame_queue.full():
            result_frame_queue.put(cv2.cvtColor(cv_image, cv2.COLOR_BGRA2BGR))

        if sleep > 0:
            time.sleep(sleep)
