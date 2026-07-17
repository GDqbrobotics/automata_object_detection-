import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from .config import ARTIFACT_COLORS, DEFAULT_ARUCO_CONFIG, DEFAULT_GRID_CONFIG

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
# Extra pixels added around the grid in the rectified crop sent to the inference
# node, so fragments touching the grid border are not cut.
CROP_MARGIN_PX = 40
# Send a new crop only when a grid corner moved more than this (pixels), so the
# crop does not wobble at every smoothing tick and make the segmentation flicker.
CROP_CHANGE_PX = 15
# After this many consecutive frames with no ArUco marker at all, the mat is
# considered removed: the grid is forgotten and the inference node is told, so
# it can reset the artifact ids for the next batch. At ~15 fps this is a few
# seconds - long enough that the robot arm briefly covering the markers can
# never trigger it.
GRID_LOST_FRAMES = 45
# The latest images (already encoded as JPEG) that the live streams send to the
# browser: one for the ArUco grid view, one for the inference view.
latest_jpeg = {"aruco": None, "inference": None}
jpeg_lock = threading.Lock()
# The latest artifact info (id, cells, grasp pose) as encoded JSON, polled by the page.
latest_info = {"json": b'{"units": "", "artifacts": []}'}
info_lock = threading.Lock()

# Daily record file: one text block per examination session (a mat placed with
# its artefacts on it), separated by dashed lines, with a daily summary at the
# top. The current session appears in the file too, marked "(in progress)", and
# is finalized when the mat is removed - so the last session of the day is in
# the file even if the app is stopped with the mat still on the table.
RECORDS_FILE = "artefact_records.txt"
RECORDS_HEADER_LINE = "=" * 50
# The records state, shared with the web server thread (the clear button), so
# every access goes through the lock like the other shared data above.
records = {
    "history": "",     # finalized session blocks, as already formatted text
    "sessions": 0,     # number of finalized sessions today
    "artefacts": 0,    # artefacts counted in the finalized sessions
    "current": {},     # current session: {artifact_id: {"cells": ..., "pose": ...}}
    "started_at": "",  # when the current session's first artefact appeared
    "units": "",       # "mm" or "m", same as the web page
}
records_lock = threading.Lock()

# The web page served at "/": the two videos side by side, and the artifact
# info panel (cells + grasp poses) at the bottom left, refreshed twice a second.
PAGE_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Automata Vision</title>
<style>
  body { font-family: system-ui, sans-serif; background: #121212; color: #d6d6d6;
         margin: 0; padding: 22px 26px; }
  h2 { margin: 0 0 8px; font-size: 11px; font-weight: 500; color: #8a8a8a;
       text-transform: uppercase; letter-spacing: 0.09em; }
  .videos { display: flex; gap: 22px; align-items: flex-start; flex-wrap: wrap; }
  .videos .panel { flex: 1; min-width: 320px; }
  .videos img { display: block; width: 100%; border: 1px solid #2c2c2c; }
  #info { margin-top: 26px; max-width: 880px; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; }
  th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #242424;
           white-space: nowrap; }
  th { color: #7a7a7a; font-weight: 500; font-size: 11px;
       text-transform: uppercase; letter-spacing: 0.06em; }
  td.mono { font-family: ui-monospace, monospace; font-variant-numeric: tabular-nums;
            color: #c4c4c4; }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
         margin-right: 8px; vertical-align: middle; }
  .empty { color: #6f6f6f; font-size: 13px; }
  #clock { position: absolute; top: 20px; right: 26px; color: #6f6f6f; font-size: 12px;
           font-family: ui-monospace, monospace; font-variant-numeric: tabular-nums; }
  .info-head { display: flex; align-items: baseline; gap: 14px; margin-bottom: 8px; }
  .info-head h2 { margin: 0; }
  .muted { color: #6f6f6f; font-size: 12px; }
  #clear { margin-left: auto; background: none; border: 1px solid #333; color: #8a8a8a;
           font-size: 11px; padding: 3px 10px; cursor: pointer; }
  #clear:hover { border-color: #555; color: #c4c4c4; }
</style>
</head>
<body>
<div id="clock"></div>
<div class="videos">
  <div class="panel"><h2>Artefact Container Cells</h2><img src="/aruco"></div>
  <div class="panel"><h2>Grasping Pose</h2><img src="/inference"></div>
</div>
<div id="info">
  <div class="info-head">
    <h2>Artefacts</h2>
    <span id="daily" class="muted"></span>
    <button id="clear" title="Restart today's artefact_records.txt from zero">Clear daily log</button>
  </div>
  <div id="artifacts"><span class="empty">Waiting for data...</span></div>
</div>
<script>
function fmt(value, units) {
  return units === "mm" ? value.toFixed(0) : value.toFixed(3);
}
async function refresh() {
  try {
    const response = await fetch("/data");
    const data = await response.json();
    const units = data.units;
    const r = data.records;
    document.getElementById("daily").textContent =
      r ? "Today: " + r.examinations + " examinations · " + r.artefacts + " artefacts" : "";
    const box = document.getElementById("artifacts");
    if (!data.artifacts.length) {
      box.innerHTML = "<span class='empty'>No artefacts on the table.</span>";
      return;
    }
    let html = "<table><tr><th>Id</th><th>Cells</th>" +
               "<th>Grasp 1 &mdash; x, y, z (" + units + ")</th>" +
               "<th>Grasp 2 &mdash; x, y, z (" + units + ")</th></tr>";
    for (const a of data.artifacts) {
      const p = a.pose;
      const g1 = p ? fmt(p.x_1, units) + ", " + fmt(p.y_1, units) + ", " + fmt(p.z_1, units) : "waiting for depth";
      const g2 = p ? fmt(p.x_2, units) + ", " + fmt(p.y_2, units) + ", " + fmt(p.z_2, units) : "";
      html += "<tr><td><span class='dot' style='background:" + a.color + "'></span>" + a.id + "</td>" +
              "<td>" + (a.cells || "-") + "</td><td class='mono'>" + g1 + "</td><td class='mono'>" + g2 + "</td></tr>";
    }
    html += "</table>";
    box.innerHTML = html;
  } catch (e) {
    // Server briefly unavailable: keep the last table and retry.
  }
}
setInterval(refresh, 500);
refresh();
function tick() {
  document.getElementById("clock").textContent = new Date().toLocaleTimeString();
}
setInterval(tick, 1000);
tick();
document.getElementById("clear").onclick = async function () {
  if (!confirm("Clear today's records? artefact_records.txt restarts from zero.")) return;
  try { await fetch("/clear_history", {method: "POST"}); } catch (e) {}
  refresh();
};
</script>
</body>
</html>
"""


class MJPEGHandler(BaseHTTPRequestHandler):
    """Serve the web page, the two MJPEG streams and the artifact info JSON."""

    def do_GET(self):
        if self.path == "/":
            body = PAGE_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/aruco":
            self.send_mjpeg("aruco")
        elif self.path == "/inference":
            self.send_mjpeg("inference")
        elif self.path == "/data":
            with info_lock:
                body = latest_info["json"]
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/clear_history":
            # The web page's clear button: restart the daily records from zero.
            # If a mat is on the table right now, its session simply starts
            # again as Examination #1 on the next update.
            with records_lock:
                records["history"] = ""
                records["sessions"] = 0
                records["artefacts"] = 0
                records["current"] = {}
                records["started_at"] = ""
            write_records_file()
            print("[ARUCO] Daily records cleared from the web page")
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def send_mjpeg(self, name):
        """Send the latest frames of one stream as an endless MJPEG video."""
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                with jpeg_lock:
                    jpeg = latest_jpeg[name]
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


def publish_to_stream(image, name="aruco"): #Encode the image as JPEG and give it to one stream
    ok, buffer = cv2.imencode(".jpg", image)
    if ok:
        with jpeg_lock:
            latest_jpeg[name] = buffer.tobytes()


def color_to_css(artifact_id):
    """Return the artifact's palette color as a CSS string like '#rrggbb'."""
    blue, green, red = ARTIFACT_COLORS[artifact_id % len(ARTIFACT_COLORS)]
    return "#%02x%02x%02x" % (red, green, blue)


def publish_info(objects, cells_by_id, camera_type):
    """Refresh the JSON the web page polls: id, color, cells and grasp pose per artifact."""
    artifacts = []
    for item in sorted(objects or [], key=lambda entry: entry["id"]):
        artifact_id = item["id"]
        occupied = cells_by_id.get(artifact_id, {})
        names = [cell_name(col, row) for (col, row) in sorted(occupied.keys(), key=lambda rc: (rc[1], rc[0]))]
        artifacts.append({
            "id": artifact_id,
            "color": color_to_css(artifact_id),
            "cells": " ".join(names),
            "pose": item.get("pose"),
        })
    # Daily counters for the web page: sessions and artefacts, including the
    # session still in progress.
    with records_lock:
        examinations = records["sessions"] + (1 if records["current"] else 0)
        total_artefacts = records["artefacts"] + len(records["current"])
    body = json.dumps({
        "units": "mm" if camera_type == "orbbec" else "m",
        "artifacts": artifacts,
        "records": {"examinations": examinations, "artefacts": total_artefacts},
    }).encode("utf-8")
    with info_lock:
        latest_info["json"] = body


def load_records_history():
    """Reload today's already saved sessions from the records file.

    Called once at startup: a restart of the app keeps counting from where it
    was. The file only restarts from zero with the web page's clear button.
    """
    try:
        with open(RECORDS_FILE, "r") as records_file:
            content = records_file.read()
    except OSError:
        return
    # The saved sessions are everything below the summary header.
    if RECORDS_HEADER_LINE in content:
        history = content.split(RECORDS_HEADER_LINE, 1)[1].lstrip("\n")
    else:
        history = content
    with records_lock:
        records["history"] = history
        records["sessions"] = history.count("Examination #")
        records["artefacts"] = sum(1 for line in history.splitlines() if line.startswith("id "))


def format_grasp_text(pose, units):
    """The two grasp points as one text line, like the web table shows them."""
    if pose is None:
        return "grasp: waiting for depth"
    number = "%.0f" if units == "mm" else "%.3f"
    grasp_1 = ", ".join(number % pose[key] for key in ("x_1", "y_1", "z_1"))
    grasp_2 = ", ".join(number % pose[key] for key in ("x_2", "y_2", "z_2"))
    return "grasp 1: %s | grasp 2: %s" % (grasp_1, grasp_2)


def format_session_block(number, started_at, session, units, in_progress):
    """One examination session as text: dashed title, then one line per artefact."""
    state = " (in progress)" if in_progress else ""
    lines = [
        "-" * 50,
        "Examination #%d - %s%s" % (number, started_at, state),
        "artefacts: %d | coordinates in %s" % (len(session), units),
        "-" * 50,
    ]
    for artifact_id in sorted(session.keys()):
        entry = session[artifact_id]
        cells = entry["cells"] if entry["cells"] else "-"
        lines.append("id %d | cells: %s | %s" % (artifact_id, cells, format_grasp_text(entry["pose"], units)))
    return "\n".join(lines) + "\n"


def write_records_file():
    """Rewrite the whole records file: summary, saved sessions, current session.

    The file is small, so rewriting it completely (safely, via a temp file) is
    the simplest way to keep the summary on top always up to date.
    """
    with records_lock:
        history = records["history"]
        sessions = records["sessions"]
        artefacts = records["artefacts"]
        current = dict(records["current"])
        started_at = records["started_at"]
        units = records["units"]

    if current:
        sessions += 1
        artefacts += len(current)

    content = "ARTEFACT RECORDS - daily log\n"
    content += "Examinations today: %d | Artefacts examined today: %d\n" % (sessions, artefacts)
    content += RECORDS_HEADER_LINE + "\n\n"
    content += history
    if current:
        content += format_session_block(sessions, started_at, current, units, True)

    temp_path = RECORDS_FILE + ".tmp"
    with open(temp_path, "w") as records_file:
        records_file.write(content)
    os.replace(temp_path, RECORDS_FILE)


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


def grid_corners_px(transform, n_cols, n_rows):
    """Return the 4 grid corners in image pixels: TL, TR, BR, BL."""
    return [
        apply_affine(transform, 0, 0),
        apply_affine(transform, n_cols, 0),
        apply_affine(transform, n_cols, n_rows),
        apply_affine(transform, 0, n_rows),
    ]


def build_crop_message(transform, n_cols, n_rows):
    """Build the rectified-crop message for the inference node.

    The crop is a straightened image of the grid only (plus CROP_MARGIN_PX on
    every side), whatever the grid's rotation in the camera image. This way the
    inference node never sees the table around the mat or the marker sheets —
    with a plain bounding-box crop, a tilted grid let big background wedges into
    the corners and BiRefNet segmented those instead of the fragments.

    The message carries the affine matrix that maps CROP pixels back to
    FULL-FRAME pixels, so the inference node can both warp the frame and convert
    its results back. Returns None if the grid is degenerate (glitch).
    """
    # Size of one grid cell in image pixels (kept, so the crop has ~native
    # resolution). The transform columns are the image-space vectors of one
    # grid step along x and along y.
    size_x = (transform[0][0] ** 2 + transform[1][0] ** 2) ** 0.5
    size_y = (transform[0][1] ** 2 + transform[1][1] ** 2) ** 0.5
    cell_px = (size_x + size_y) / 2.0
    if cell_px < 10:
        return None

    grid_w = int(round(n_cols * cell_px))
    grid_h = int(round(n_rows * cell_px))
    margin = CROP_MARGIN_PX

    # 3 matching points, crop pixels -> full-frame pixels (3 grid corners).
    crop_points = np.array([
        [margin, margin],
        [margin + grid_w, margin],
        [margin, margin + grid_h],
    ], dtype=np.float32)
    image_points = np.array([
        apply_affine(transform, 0, 0),
        apply_affine(transform, n_cols, 0),
        apply_affine(transform, 0, n_rows),
    ], dtype=np.float32)
    matrix = cv2.getAffineTransform(crop_points, image_points)

    return {
        "matrix": matrix,
        "width": grid_w + 2 * margin,
        "height": grid_h + 2 * margin,
        "margin": margin,
    }


def grid_corners_moved(old_corners, new_corners):
    """True when any grid corner moved more than CROP_CHANGE_PX pixels."""
    if old_corners is None:
        return True
    for old, new in zip(old_corners, new_corners):
        dx = new[0] - old[0]
        dy = new[1] - old[1]
        if (dx * dx + dy * dy) ** 0.5 > CROP_CHANGE_PX:
            return True
    return False


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


def draw_heatmap(image, transform, cells_by_id):
    """Color the occupied cells. Each artifact keeps its own color (chosen from its
    stable id) and a cell is drawn stronger the more it is covered (heatmap).
    """
    for artifact_id, occupied in cells_by_id.items():
        color = ARTIFACT_COLORS[artifact_id % len(ARTIFACT_COLORS)]
        for (col, row), coverage in occupied.items():
            # The 4 corners of the cell in image pixels.
            p1 = apply_affine(transform, col - 1, row - 1)
            p2 = apply_affine(transform, col, row - 1)
            p3 = apply_affine(transform, col, row)
            p4 = apply_affine(transform, col - 1, row)
            poly = np.array([p1, p2, p3, p4], dtype=np.int32)
            fill_cell(image, poly, color, MAX_ALPHA * coverage)


def start_aruco(*, frame_queue, parameters_queue, objects_queue=None, crop_queue=None, result_frame_queue=None, verbose=False, sleep=0.0, camera_type="realsense"):
    """ArUco node: read color frames, detect the markers, draw the grid overlay.

    It serves a web page at http://localhost:8090 with two live videos (the
    grid view and the inference view) plus a realtime panel with each artifact's
    occupied cells and grasp pose, and saves aruco_result.png every few seconds.
    It also colors the grid cells occupied by each artifact (using the outlines
    sent by the inference node). It does not use depth yet.

    Args:
        frame_queue: queue with the color images from the camera.
        parameters_queue: camera intrinsics (not used yet, needed later for 3D).
        objects_queue: [{"id": artifact_id, "contour": outline, "pose": ...}, ...]
            from the inference node, full-frame coords, tagged with the artifact's
            stable id so a cell's color/occupancy stays linked to one physical
            artifact; the pose is shown on the web page.
        crop_queue: where to send the grid's rectified crop so the inference
            node's crop follows the grid instead of being static.
        result_frame_queue: annotated frames from the inference node (what
            result.png shows), served as the second video on the web page.
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

    # Start the web server (page + the two video streams + artifact info).
    start_stream_server(STREAM_PORT)
    print("[ARUCO] Live view available at http://localhost:%d" % STREAM_PORT)

    # Reload today's records so a restart keeps counting from where it was.
    load_records_history()

    n_cols = DEFAULT_GRID_CONFIG.n_cols
    n_rows = DEFAULT_GRID_CONFIG.n_rows

    grid = None          # last good grid corners (kept so the overlay stays stable)
    skipped = 0          # how many frames in a row we ignored because of a big jump
    frames_without_markers = 0  # consecutive frames with no marker detected at all
    grid_lost = False           # True after the mat was judged removed (no markers for a while)
    last_snapshot = 0.0
    objects = None            # latest [{"id":.., "contour":..}] from inference (full-frame)
    objects_dirty = False     # True when a new outlines payload arrived
    force_recompute = False   # True when the grid itself really moved: recompute every id's cells
    cells_by_id = {}          # cached: {artifact_id: {(col, row): coverage}}
    last_sent_corners = None  # grid corners (image px) of the last crop sent to inference

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

            # Pick up the latest annotated inference frame and show it as the
            # second video on the web page.
            if result_frame_queue is not None and not result_frame_queue.empty():
                publish_to_stream(result_frame_queue.get(), "inference")

            # Detect the markers on the gray version of the image.
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)

            image = frame.copy()
            if ids is not None:
                # At least one marker is visible, so the mat is there.
                frames_without_markers = 0
                grid_lost = False

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
                        # This is a real change, so every artifact's cells must be
                        # recomputed against the new grid.
                        grid = found
                        skipped = 0
                        force_recompute = True
                    elif grid_movement(grid, found) < JUMP_LIMIT:
                        # Normal small update: just smooth toward it. This on its
                        # own is NOT a real move, so already placed artifacts keep
                        # their cached cells (this is what keeps them from drifting
                        # every time the grid is re-detected).
                        grid = smooth_corners(grid, found)
                        skipped = 0
                    elif skipped >= MAX_SKIPPED:
                        # Big jump accepted after too many ignored frames: the grid
                        # really moved, so adopt the new position at once. Smoothing
                        # toward a far away position would crawl there a few percent
                        # per accepted frame and take many seconds to catch up.
                        grid = found
                        skipped = 0
                        force_recompute = True
                    else:
                        # Big jump: probably a glitch, ignore this frame.
                        skipped += 1
            else:
                if verbose:
                    print("[ARUCO] No markers detected")

                # No marker at all: if it lasts long enough, the mat was removed.
                # Forget the grid (the overlay disappears from the web page) and
                # tell the inference node once, so it resets the artifact ids.
                frames_without_markers += 1
                if frames_without_markers >= GRID_LOST_FRAMES and not grid_lost:
                    grid_lost = True
                    grid = None
                    skipped = 0
                    # Force an unconditional crop send when the grid comes back,
                    # so the inference node also learns the grid returned.
                    last_sent_corners = None
                    print("[ARUCO] No markers for a while - mat removed")

                    # The mat is gone: finalize the examination session in the
                    # daily records file (if any artefact was seen).
                    with records_lock:
                        if records["current"]:
                            records["sessions"] += 1
                            block = format_session_block(
                                records["sessions"], records["started_at"],
                                records["current"], records["units"], False,
                            )
                            records["history"] += block + "\n"
                            records["artefacts"] += len(records["current"])
                            records["current"] = {}
                            records["started_at"] = ""
                            print("[ARUCO] Examination #%d saved to %s" % (records["sessions"], RECORDS_FILE))
                    write_records_file()
                    if crop_queue is not None:
                        if crop_queue.full():
                            try:
                                crop_queue.get_nowait()
                            except Exception:
                                pass
                        crop_queue.put({"grid_lost": True})

            # Draw the last known grid (stays on screen even if a marker is missed).
            transform = grid_to_image_transform(grid, n_cols, n_rows)
            if transform is not None:
                draw_grid(image, transform, n_cols, n_rows)
                draw_labels(image, transform, n_cols, n_rows)

                # Tell the inference node where the grid is, so its crop follows
                # the grid. Sent only when a corner really moved. Freshest wins:
                # a stale crop the inference did not consume yet is replaced,
                # never left in the queue to be applied late.
                if crop_queue is not None:
                    corners_px = grid_corners_px(transform, n_cols, n_rows)
                    if grid_corners_moved(last_sent_corners, corners_px):
                        message = build_crop_message(transform, n_cols, n_rows)
                        if message is not None:
                            if crop_queue.full():
                                try:
                                    crop_queue.get_nowait()
                                except Exception:
                                    pass
                            crop_queue.put(message)
                            last_sent_corners = corners_px
                            if verbose:
                                print("[ARUCO] New crop sent: %dx%d px" % (message["width"], message["height"]))

                # Update which cells each artifact occupies only when needed
                # (BiRefNet is slow, so new outlines arrive rarely). An artifact
                # that is still there and whose grid did not really move keeps its
                # previously computed cells untouched - this is what keeps a placed
                # artifact from drifting or changing color every time the grid is
                # re-detected.
                if objects is not None and (objects_dirty or force_recompute):
                    current_ids = set(item["id"] for item in objects)

                    # Drop cells for artifacts no longer being tracked (removed).
                    for stale_id in list(cells_by_id.keys()):
                        if stale_id not in current_ids:
                            del cells_by_id[stale_id]

                    for item in objects:
                        # Skip artifacts already cached, unless the grid itself
                        # moved and everything must be recomputed against it.
                        if item["id"] in cells_by_id and not force_recompute:
                            continue
                        cells_by_id[item["id"]] = fragment_cells(
                            transform, item["contour"], n_cols, n_rows,
                            DEFAULT_GRID_CONFIG.occupancy_threshold,
                            DEFAULT_GRID_CONFIG.coverage_samples,
                        )

                    objects_dirty = False
                    force_recompute = False

                # Color the occupied cells every frame, following the current grid.
                draw_heatmap(image, transform, cells_by_id)

            # Keep the daily records' current session up to date: every artefact
            # seen in this session, with its latest cells and grasp pose. Not
            # while the mat is judged removed: right after that, "objects" still
            # holds the last artefacts for a moment (until the inference node's
            # reset arrives), and they must not restart a session by mistake.
            if objects and not grid_lost:
                with records_lock:
                    records["units"] = "mm" if camera_type == "orbbec" else "m"
                    if not records["current"]:
                        records["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    for item in objects:
                        occupied = cells_by_id.get(item["id"], {})
                        names = [cell_name(col, row) for (col, row) in sorted(occupied.keys(), key=lambda rc: (rc[1], rc[0]))]
                        records["current"][item["id"]] = {"cells": " ".join(names), "pose": item.get("pose")}

            # Send the image to the live stream every frame, and refresh the
            # info panel (cells + grasp poses) the web page polls.
            publish_to_stream(image, "aruco")
            publish_info(objects, cells_by_id, camera_type)

            # Save a snapshot to disk only every few seconds.
            now = time.time()
            if now - last_snapshot >= SNAPSHOT_INTERVAL_S:
                save_image(image, "aruco_result.png")
                # Keep the records file fresh too (the current session is in it,
                # marked "in progress", so nothing is lost if the app stops).
                write_records_file()
                last_snapshot = now
                # Print the occupied cells for each artifact (no percentages),
                # sorted by id so the same artifact always prints on the same line.
                for artifact_id in sorted(cells_by_id.keys()):
                    occupied = cells_by_id[artifact_id]
                    names = [cell_name(col, row) for (col, row) in sorted(occupied.keys(), key=lambda rc: (rc[1], rc[0]))]
                    print("[ARUCO] Reperto %d: %s" % (artifact_id, " ".join(names)))
        except Exception as error:
            # Log the problem and keep going with the next frame.
            print("[ARUCO] Skipping a frame after an error:", error)

        if sleep > 0:
            time.sleep(sleep)
