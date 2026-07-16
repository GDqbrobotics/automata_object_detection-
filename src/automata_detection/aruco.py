import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from .config import DEFAULT_ARUCO_CONFIG, DEFAULT_GRID_CONFIG

# --- Settings (easy to change) ---
STREAM_PORT = 8090            # open http://localhost:8090 in a browser to watch the video
SNAPSHOT_INTERVAL_S = 5.0     # save aruco_result.png every few seconds
SMOOTHING = 0.1               # 0..1: higher follows the markers faster, lower is smoother
JUMP_LIMIT = 40               # pixels: ignore a new grid that jumps more than this (likely a glitch)
MAX_SKIPPED = 15              # accept the new grid anyway after this many ignored frames (a real move)
# Distance from the real grid edge to the marker center, measured with a ruler (in mm).
# MARGIN_X is the left/right distance, MARGIN_Y is the top/bottom distance.
# Increase a value if the red grid is too big on that axis, decrease if too small.
MARGIN_X_MM = 15
MARGIN_Y_MM = 15
# Color of the row/column labels drawn on the grid border (BGR, red like the grid).
LABEL_COLOR = (0, 0, 255)
# Strongest transparency used to color a fully covered cell (0..1). A cell that is
# only partly covered is drawn lighter, in proportion to its coverage (heatmap).
MAX_ALPHA = 0.6
# One color per fragment (BGR), cycled if there are more fragments than colors.
HEATMAP_COLORS = [
    (255, 0, 0),      # blue
    (0, 0, 255),      # red
    (0, 200, 0),      # green
    (0, 200, 255),    # yellow
    (255, 0, 255),    # magenta
    (255, 200, 0),    # cyan
    (0, 128, 255),    # orange
]
# The latest image (already encoded as JPEG) that the live stream sends to the browser.
latest_jpeg = {"data": None}
jpeg_lock = threading.Lock()


class MJPEGHandler(BaseHTTPRequestHandler): #Send the latest frame to the browser as an MJPEG video stream
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                with jpeg_lock:
                    jpeg = latest_jpeg["data"]
                if jpeg is not None:
                    self.wfile.write(b"--frame\r\n")
                    header = "Content-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpeg)
                    self.wfile.write(header.encode("ascii"))
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                time.sleep(0.03)  # about 30 frames per second
        except Exception:
            # The browser closed or reloaded the page: just stop this stream.
            pass

    def log_message(self, *args):
        # Do not print a line for every request.
        return


def start_stream_server(port): #start the video stream server in the background
    server = ThreadingHTTPServer(("0.0.0.0", port), MJPEGHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()


def publish_to_stream(image): #Encode the image as JPEG and give it to the stream
    ok, buffer = cv2.imencode(".jpg", image)
    if ok:
        with jpeg_lock:
            latest_jpeg["data"] = buffer.tobytes()


def save_image(image, path): #Save the image to disk safely (write a temp file, then rename it)
    temp_path = path + ".tmp.png"
    cv2.imwrite(temp_path, image)
    os.replace(temp_path, path)


def marker_center(marker):#Return the center (x, y) of one marker (the average of its 4 corners).
    points = marker[0]  # the 4 corner points of the marker
    x = (points[0][0] + points[1][0] + points[2][0] + points[3][0]) / 4.0
    y = (points[0][1] + points[1][1] + points[2][1] + points[3][1]) / 4.0
    return x, y


def corner_role(center_x, center_y, middle_x, middle_y): #Tell which corner a marker is, by comparing it to the middle of all markers.
    if center_x < middle_x and center_y < middle_y:
        return "TL"  # top-left
    if center_x >= middle_x and center_y < middle_y:
        return "TR"  # top-right
    if center_x >= middle_x and center_y >= middle_y:
        return "BR"  # bottom-right
    return "BL"      # bottom-left


def find_grid_corners(markers):
    """Find the grid corners from the detected markers.

    For each corner we use the AVERAGE center of the markers placed there. The
    center is much steadier than a single corner point, so the overlay shakes
    less. Several markers on the same corner (a small board) are averaged together.

    Returns a dictionary like {"TL": point, "TR": point, "BL": point}.
    """
    # Step 1: find the middle point of all the markers.
    sum_x = 0.0
    sum_y = 0.0
    for marker in markers:
        center_x, center_y = marker_center(marker)
        sum_x += center_x
        sum_y += center_y
    middle_x = sum_x / len(markers)
    middle_y = sum_y / len(markers)

    # Step 2: group the marker centers by corner and average them.
    totals = {}  # role -> [sum_x, sum_y, count]
    for marker in markers:
        center_x, center_y = marker_center(marker)
        role = corner_role(center_x, center_y, middle_x, middle_y)
        if role not in totals:
            totals[role] = [0.0, 0.0, 0]
        totals[role][0] += center_x
        totals[role][1] += center_y
        totals[role][2] += 1

    corners = {}
    for role in totals:
        total_x, total_y, count = totals[role]
        corners[role] = np.array([total_x / count, total_y / count], dtype=np.float32)
    return corners


def smooth_corners(old_corners, new_corners): #Move the old corners a little toward the new ones, to reduce shaking.
    for role in new_corners:
        old_corners[role] = SMOOTHING * new_corners[role] + (1 - SMOOTHING) * old_corners[role]
    return old_corners


def grid_movement(old_corners, new_corners): #Return the largest distance a corner moved between two grids (in pixels).
    largest = 0.0
    for role in new_corners:
        dx = new_corners[role][0] - old_corners[role][0]
        dy = new_corners[role][1] - old_corners[role][1]
        distance = (dx * dx + dy * dy) ** 0.5
        if distance > largest:
            largest = distance
    return largest


def role_grid_coord(role, n_cols, n_rows):
    """Return the grid coordinate (in cells) of a marker center for a given corner.

    The markers sit outside the grid, so their coordinates are negative or
    larger than the grid size by the margins.
    """
    margin_x = MARGIN_X_MM / DEFAULT_GRID_CONFIG.cell_size_mm
    margin_y = MARGIN_Y_MM / DEFAULT_GRID_CONFIG.cell_size_mm
    if role == "TL":
        return -margin_x, -margin_y
    if role == "TR":
        return n_cols + margin_x, -margin_y
    if role == "BR":
        return n_cols + margin_x, n_rows + margin_y
    return -margin_x, n_rows + margin_y  # "BL"


def apply_affine(transform, x, y):
    """Apply a 2x3 affine transform to a point and return integer pixels."""
    px = transform[0][0] * x + transform[0][1] * y + transform[0][2]
    py = transform[1][0] * x + transform[1][1] * y + transform[1][2]
    return int(px), int(py)


def grid_to_image_transform(corners, n_cols, n_rows):
    """Build the affine transform that maps grid-cell coordinates to image pixels.

    Uses the 3 known marker positions. Returns a 2x3 transform, or None if we do
    not have exactly 3 corners yet.
    """
    if corners is None or len(corners) != 3:
        return None
    # Build 3 matching points: grid coordinate -> marker center in the image.
    grid_points = []
    image_points = []
    for role in corners:
        grid_x, grid_y = role_grid_coord(role, n_cols, n_rows)
        grid_points.append([grid_x, grid_y])
        image_points.append([corners[role][0], corners[role][1]])

    grid_points = np.array(grid_points, dtype=np.float32)
    image_points = np.array(image_points, dtype=np.float32)
    return cv2.getAffineTransform(grid_points, image_points)


def draw_grid(image, transform, n_cols, n_rows):
    """Draw the n_cols x n_rows grid lines using the grid->image transform."""
    # Vertical lines (one per column boundary).
    for c in range(n_cols + 1):
        point1 = apply_affine(transform, c, 0)
        point2 = apply_affine(transform, c, n_rows)
        cv2.line(image, point1, point2, (0, 0, 255), 1)

    # Horizontal lines (one per row boundary).
    for r in range(n_rows + 1):
        point1 = apply_affine(transform, 0, r)
        point2 = apply_affine(transform, n_cols, r)
        cv2.line(image, point1, point2, (0, 0, 255), 1)


def draw_labels(image, transform, n_cols, n_rows):
    """Write the column numbers (1..n_cols) above the top edge and the row
    letters (A, B, ...) to the left of the grid, using the grid transform.
    """
    # Column numbers along the top border.
    for c in range(1, n_cols + 1):
        px, py = apply_affine(transform, c - 0.5, -0.4)
        cv2.putText(image, str(c), (px - 6, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, LABEL_COLOR, 1, cv2.LINE_AA)

    # Row letters along the left border.
    for r in range(1, n_rows + 1):
        letter = chr(ord("A") + r - 1)
        px, py = apply_affine(transform, -0.5, r - 0.5)
        cv2.putText(image, letter, (px - 6, py + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, LABEL_COLOR, 1, cv2.LINE_AA)


def cell_name(col, row):
    """Return a cell name like 'C3' from 1-based column and row numbers."""
    letter = chr(ord("A") + row - 1)
    return "%s%d" % (letter, col)


def cell_coverage(transform, contour, col, row, samples):
    """Return the fraction (0..1) of cell (col, row) covered by the fragment.

    We sample a samples x samples grid of points inside the cell, map them to
    image pixels and count how many fall inside the fragment outline.
    """
    inside = 0
    total = 0
    for i in range(samples):
        for j in range(samples):
            # A point inside the cell, in grid coordinates. Cell (col, row) spans
            # the grid square [col-1, col] x [row-1, row].
            grid_x = (col - 1) + (i + 0.5) / samples
            grid_y = (row - 1) + (j + 0.5) / samples
            px, py = apply_affine(transform, grid_x, grid_y)
            if cv2.pointPolygonTest(contour, (px, py), False) >= 0:
                inside += 1
            total += 1
    return inside / total


def fragment_cells(transform, contour, n_cols, n_rows, threshold, samples):
    """Return a dict {(col, row): coverage} for the cells this fragment occupies."""
    occupied = {}
    for col in range(1, n_cols + 1):
        for row in range(1, n_rows + 1):
            coverage = cell_coverage(transform, contour, col, row, samples)
            if coverage >= threshold:
                occupied[(col, row)] = coverage
    return occupied


def fill_cell(image, poly, color, alpha):
    """Blend a filled polygon onto the image with the given alpha (0..1).

    We only work on the polygon's bounding box, so coloring many cells stays fast.
    """
    x, y, w, h = cv2.boundingRect(poly)
    # Clip the bounding box to the image, just in case.
    x0 = max(x, 0)
    y0 = max(y, 0)
    x1 = min(x + w, image.shape[1])
    y1 = min(y + h, image.shape[0])
    if x1 <= x0 or y1 <= y0:
        return

    roi = image[y0:y1, x0:x1]
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    shifted = (poly - [x0, y0]).astype(np.int32)
    cv2.fillConvexPoly(mask, shifted, 255)

    color_layer = np.zeros_like(roi)
    color_layer[:] = color
    blended = cv2.addWeighted(roi, 1 - alpha, color_layer, alpha, 0)
    # Replace only the pixels inside the cell polygon.
    roi[mask > 0] = blended[mask > 0]


def draw_heatmap(image, transform, fragments_cells):
    """Color the occupied cells. Each fragment has its own color and a cell is
    drawn stronger the more it is covered (heatmap).
    """
    for index, occupied in enumerate(fragments_cells):
        color = HEATMAP_COLORS[index % len(HEATMAP_COLORS)]
        for (col, row), coverage in occupied.items():
            # The 4 corners of the cell in image pixels.
            p1 = apply_affine(transform, col - 1, row - 1)
            p2 = apply_affine(transform, col, row - 1)
            p3 = apply_affine(transform, col, row)
            p4 = apply_affine(transform, col - 1, row)
            poly = np.array([p1, p2, p3, p4], dtype=np.int32)
            fill_cell(image, poly, color, MAX_ALPHA * coverage)


def start_aruco(*, frame_queue, parameters_queue, objects_queue=None, verbose=False, sleep=0.0, camera_type="realsense"):
    """ArUco node: read color frames, detect the markers, draw the grid overlay.

    It shows a live video at http://localhost:8090 and saves aruco_result.png
    every few seconds. It also colors the grid cells occupied by each fragment
    (using the outlines sent by the inference node). It does not use depth yet.

    Args:
        frame_queue: queue with the color images from the camera.
        parameters_queue: camera intrinsics (not used yet, needed later for 3D).
        objects_queue: fragment outlines (full-frame coords) from the inference node.
        verbose: print debug information.
        sleep: time to wait between iterations.
        camera_type: "realsense" or "orbbec".
    """
    # Build the ArUco detector for our dictionary.
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DEFAULT_ARUCO_CONFIG.dictionary))
    detector_params = cv2.aruco.DetectorParameters()
    # Refine the marker corners to sub-pixel precision: this makes the detected
    # corners much steadier and reduces the small jumps frame to frame.
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, detector_params)

    # Start the live video stream.
    start_stream_server(STREAM_PORT)
    print("[ARUCO] Live stream available at http://localhost:%d" % STREAM_PORT)

    n_cols = DEFAULT_GRID_CONFIG.n_cols
    n_rows = DEFAULT_GRID_CONFIG.n_rows

    grid = None          # last good grid corners (kept so the overlay stays stable)
    skipped = 0          # how many frames in a row we ignored because of a big jump
    last_snapshot = 0.0
    objects = None            # latest fragment outlines (full-frame) from inference
    objects_dirty = False     # True when new outlines arrived and cells must be recomputed
    fragments_cells = []      # cached: a {(col, row): coverage} dict per fragment

    while True:
        if frame_queue.empty():
            time.sleep(sleep)
            continue

        # Process one frame. Any unexpected error here is caught below so a
        # single bad frame can never kill the process (and freeze the stream).
        try:
            # Get the color image (no depth for now).
            frame = frame_queue.get()

            # Pick up the latest fragment outlines from the inference node (if any).
            if objects_queue is not None and not objects_queue.empty():
                objects = objects_queue.get()
                objects_dirty = True

            # Detect the markers on the gray version of the image.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)

            image = frame.copy()
            if ids is not None:
                # Draw only the marker outlines, not the ID numbers (pass None
                # instead of ids so OpenCV does not print the id over the marker).
                cv2.aruco.drawDetectedMarkers(image, corners, None)
                if verbose:
                    print("[ARUCO] Detected markers:", ids.flatten().tolist())

                # Group the markers into grid corners (works also if there are more
                # than 3 markers, for example a small board in each corner).
                found = find_grid_corners(corners)
                if verbose:
                    print("[ARUCO] Corners found:", list(found.keys()))

                # We use the grid only when we found exactly 3 corners (one is empty).
                if len(found) == 3:
                    if grid is None or set(found.keys()) != set(grid.keys()):
                        # First grid, or the set of detected corners changed (for
                        # example a marker was occluded and another reappeared):
                        # take the new grid as-is instead of comparing it. Comparing
                        # two grids with different corners would fail, so we just
                        # adopt the new one and start smoothing again from here.
                        grid = found
                        skipped = 0
                    elif grid_movement(grid, found) < JUMP_LIMIT or skipped >= MAX_SKIPPED:
                        # Normal update (or accept after too many ignored frames).
                        grid = smooth_corners(grid, found)
                        skipped = 0
                    else:
                        # Big jump: probably a glitch, ignore this frame.
                        skipped += 1
            elif verbose:
                print("[ARUCO] No markers detected")

            # Draw the last known grid (stays on screen even if a marker is missed).
            transform = grid_to_image_transform(grid, n_cols, n_rows)
            if transform is not None:
                draw_grid(image, transform, n_cols, n_rows)
                draw_labels(image, transform, n_cols, n_rows)

                # Recompute which cells each fragment occupies only when new
                # outlines arrived (BiRefNet is slow, so this happens rarely).
                if objects is not None and objects_dirty:
                    fragments_cells = []
                    for contour in objects:
                        occupied = fragment_cells(
                            transform, contour, n_cols, n_rows,
                            DEFAULT_GRID_CONFIG.occupancy_threshold,
                            DEFAULT_GRID_CONFIG.coverage_samples,
                        )
                        fragments_cells.append(occupied)
                    objects_dirty = False

                # Color the occupied cells every frame, following the current grid.
                draw_heatmap(image, transform, fragments_cells)

            # Send the image to the live stream every frame.
            publish_to_stream(image)

            # Save a snapshot to disk only every few seconds.
            now = time.time()
            if now - last_snapshot >= SNAPSHOT_INTERVAL_S:
                save_image(image, "aruco_result.png")
                last_snapshot = now
                # Print the occupied cells for each fragment (no percentages).
                for index, occupied in enumerate(fragments_cells):
                    names = [cell_name(col, row) for (col, row) in sorted(occupied.keys(), key=lambda rc: (rc[1], rc[0]))]
                    print("[ARUCO] Reperto %d: %s" % (index + 1, " ".join(names)))
        except Exception as error:
            # Log the problem and keep going with the next frame.
            print("[ARUCO] Skipping a frame after an error:", error)

        if sleep > 0:
            time.sleep(sleep)
