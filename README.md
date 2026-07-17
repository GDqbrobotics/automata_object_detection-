# Automata Object Detection

Vision pipeline for detecting archaeological fragments (ceramics, lithics) on a
work surface using a depth camera and the [BiRefNet](https://github.com/ZhengPeng7/BiRefNet)
segmentation model.

For every detected object the pipeline:

1. captures aligned **RGB + depth** frames,
2. segments the object with BiRefNet and extracts its contour,
3. computes the **shortest segment passing through the contour centroid**
   (the natural grasp axis of the fragment),
4. converts the two endpoints of that segment into **3D physical coordinates**
   using the camera intrinsics,
5. publishes those coordinates over **MQTT** and writes an annotated
   `result.png` for inspection.

Two depth cameras are supported through a common interface:

- **Intel RealSense** (D415)
- **Orbbec** (e.g. Astra 2 / Gemini)

---

## Architecture

The application runs three processes that communicate through
`multiprocessing.Queue` objects:

```
                 frame_queue            send_queue
  [ CAMERA ] ──────────────▶ [ INFERENCE ] ──────────▶ [ MQTT ]
      │                            ▲
      │      parameters_queue      │
      └────────────────────────────┘   (Orbbec intrinsics, sent once)
```

| Process | Module | Responsibility |
|---------|--------|----------------|
| Camera | `camera.py` (RealSense) / `camera_orbbec.py` (Orbbec) | Capture RGB + depth, align the streams, apply a temporal filter, push frames into `frame_queue`. |
| Inference | `inference.py` + `model.py` + `pose.py` | Crop the region of interest, run BiRefNet, find contours, compute grasp segments, convert to 3D, write `result.png`. |
| MQTT | `mqtt.py` | Read pose messages from `send_queue` and publish them as JSON to the configured topic. |

`frame_queue` and `parameters_queue` have `maxsize=1`, so the inference stage
always works on the most recent frame (older frames are dropped instead of
queuing up).

### Source layout

```
object_extraction.py            Thin entrypoint (must be run from the repo root).
src/automata_detection/
├── __init__.py                 Sets up sys.path so the BiRefNet submodule imports work.
├── __main__.py                 Allows `python -m automata_detection`.
├── cli.py                      Argument parsing and process orchestration.
├── config.py                   Fixed camera intrinsics and crop region.
├── camera.py                   RealSense driver.
├── camera_orbbec.py            Orbbec driver (same read_camera() signature).
├── model.py                    BiRefNet loading and segmentation.
├── inference.py                Contour extraction and result generation.
├── pose.py                     Grasp-segment geometry and pixel→3D conversion.
└── mqtt.py                     MQTT publisher.
BiRefNet/                       Git submodule with the segmentation model.
mqtt/                           Bundled Mosquitto broker config and credentials.
Dockerfile / Dockerfile.orbbec  Container images for each camera.
docker-compose*.yml             Orchestration (camera app + Mosquitto broker).
```

---

## Installation

### 1. Clone the repository (with submodules)

BiRefNet is a Git **submodule**, so it must be fetched too. Clone recursively:

```bash
git clone --recurse-submodules https://github.com/GDqbrobotics/automata_object_detection-.git
cd automata_object_detection-
```

If you already cloned without `--recurse-submodules`, initialize the submodule
afterwards:

```bash
git submodule update --init --recursive
```

To later pull updates including the submodule:

```bash
git pull --recurse-submodules
```

### 2. Requirements

- Python **>= 3.10**
- A CUDA-capable NVIDIA GPU is **recommended** (BiRefNet runs in half precision
  on CUDA). The pipeline also runs on CPU automatically if no GPU is available,
  but much slower.
- The driver for the camera you intend to use:
  - RealSense → `pyrealsense2`
  - Orbbec → `pyorbbecsdk`

### 3. Install Python dependencies

```bash
python3 -m pip install -r requirements.txt
```

Or install the package itself (editable), which also makes
`python -m automata_detection` available:

```bash
python3 -m pip install -e .
```

> Both camera drivers (`pyrealsense2` and `pyorbbecsdk`) are listed as
> dependencies. If you only use one camera you can install the corresponding
> driver only — the unused driver is imported lazily and is not required at
> import time.

---

## Switching between cameras

The camera is selected at runtime with `--camera-type`. The CLI imports the
matching driver module accordingly, so you never need to edit code to switch.

**RealSense (default):**

```bash
python3 object_extraction.py --verbose
```

**Orbbec:**

```bash
python3 object_extraction.py --camera-type orbbec --stream-width 1280 --stream-height 720 --verbose
```

Differences between the two cameras:

| | RealSense | Orbbec |
|---|-----------|--------|
| Driver package | `pyrealsense2` | `pyorbbecsdk` |
| Color stream resolution | `--stream-width` × `--stream-height` (default 1920×1080) | tries to match `--stream-width` × `--stream-height` (use 1280×720) |
| Depth stream | fixed 1280×720 @ 15 fps | camera default profile, rescaled to color |
| Intrinsics | **hard-coded** in `config.py` | **read from the device at runtime** and passed to inference via `parameters_queue` |
| Extra image settings | saturation/contrast/exposure forced in code | — |
| Depth post-processing | temporal filter | temporal filter + depth clamped to 20–10000 mm + nearest-point search |

> Run the application **from the repository root**: `object_extraction.py`
> imports `src.automata_detection.cli`, which only resolves when the current
> directory is the project root.

---

## Running the pipeline (local)

1. Make sure an MQTT broker is reachable (see [MQTT](#mqtt-configuration); you
   can use the bundled Mosquitto broker via Docker, or your own).
2. Plug in the camera.
3. Start the app:

```bash
# RealSense, default settings
python3 object_extraction.py --verbose

# Orbbec, custom broker
python3 object_extraction.py \
    --camera-type orbbec \
    --stream-width 1280 --stream-height 720 \
    --mqtt-host 192.168.1.50 --mqtt-port 1883 \
    --mqtt-user mqtt --mqtt-password secret \
    --mqtt-send-topic test_coordinate \
    --verbose
```

While running, the app continuously overwrites `result.png` in the working
directory with the annotated frame (contours, centroids, grasp segment, and a
timestamp) and publishes one MQTT message per processed frame that contains at
least one valid object.

---

## Command-line options

All options have defaults, so the app can be launched with no arguments.

| Option | Default | Description |
|--------|---------|-------------|
| `--camera-type {realsense,orbbec}` | `realsense` | Depth camera to use. |
| `--stream-width WIDTH` | `1920` | RGB stream width. Use `1280` for Orbbec. |
| `--stream-height HEIGHT` | `1080` | RGB stream height. Use `720` for Orbbec. |
| `--mqtt-host HOST` | `192.168.139.70` | MQTT broker host. |
| `--mqtt-port PORT` | `1883` | MQTT broker port. |
| `--mqtt-user USER` | `mqtt` | MQTT username. |
| `--mqtt-password PASS` | (set in code) | MQTT password. |
| `--mqtt-send-topic TOPIC` | `test_coordinate` | Topic the pose messages are published to. |
| `--inference-sleep SEC` | `0.01` | Sleep between inference iterations (and when the frame queue is empty). |
| `--verbose` | off | Print detailed `[CAMERA]` / `[INFERENCE]` / `[MQTT]` debug output. |

> **Security note:** the MQTT host, username and password ship with non-empty
> default values baked into `cli.py`. Always override them on the command line
> (or change the defaults) for any real deployment, and do not rely on the
> committed credentials.

---

## Fixed configuration (`config.py`)

Some parameters are not exposed on the command line and live in
`src/automata_detection/config.py` (`CameraConfig`):

| Field | Value | Meaning |
|-------|-------|---------|
| `width` / `height` | `1280` / `720` | Depth reference resolution. |
| `K` | 3×3 intrinsic matrix (fx, fy, cx, cy) | Intrinsics used by the **RealSense** pixel→3D conversion. |
| `D` | zeros | Distortion coefficients (none). |
| `crop_width` / `crop_height` | `720` / `420` | Size of the region of interest that is segmented (initial value). |
| `crop_starting_row` / `crop_starting_col` | centered in 1280×720 | Top-left corner of the crop (initial value). |

Only the area inside this crop is processed, which keeps inference fast and
limits detection to the working surface. These values are only the **starting
crop**: as soon as the ArUco node detects the grid, it sends a **rectified crop**
(the grid straightened via an affine warp, plus a margin) to the inference node
and the crop follows the grid automatically from then on — re-aligning whenever
the grid or the camera really moves, and staying correct even when the mat is
rotated on the table.

Other tunables in `config.py` (not CLI flags):

- **Detection filters** (`DetectionConfig`): `alpha_threshold` (default `250`,
  mask pixels at or above it count as object — phantom masks are usually
  weaker), `min_contour_size` / `max_contour_size` (default `200` / `4000`,
  contours outside this range are rejected as noise or oversized blobs).
- **Depth filters** (`DepthFilterConfig`): `max_artifact_height_mm`
  (default `40`) — a segmented blob that sticks out of the table more than this
  is treated as a hand or robot arm, not a fragment, and is ignored;
  `min_artifact_height_mm` (default `2.0`) — a blob that does not rise above its
  local surroundings by at least this much is treated as mat texture/shadow (a
  BiRefNet "phantom") and never confirmed; set to `0` to disable, e.g. if very
  flat fragments get rejected; `occlusion_freeze_fraction` (default `0.10`) —
  when more than this fraction of the crop's depth pixels is "tall", the scene
  counts as occluded and artifact tracking freezes (see below); `table_alpha` /
  `table_relearn_cycles` — how the estimated table depth follows the
  measurements.
- **Artifact tracking** (`TrackerConfig`): `match_distance_px` (default `50`,
  max centroid movement between cycles still counted as the same artifact),
  `confirm_hits` (default `3`, consecutive detections a new artifact needs
  before it gets an id and is published — kills one-cycle phantoms),
  `miss_limit` (default `15`, consecutive cycles an artifact can go undetected
  before it is dropped), `reacquire_window_s` (default `20.0`, how long a
  dropped artifact can still get its old id back if it reappears at the same
  spot), `heartbeat_s` (default `5.0`, how often the full artifact list is
  republished even with no changes). See below.

Tunables that still live in the code:

- **Depth range** for Orbbec in `camera_orbbec.py`: `MIN_DEPTH = 20 mm`,
  `MAX_DEPTH = 10000 mm`.
- **BiRefNet input size**: `1024×1024` in `model.py`.
- **Crop margin around the grid** and ArUco stream/smoothing settings at the top
  of `aruco.py`, including `GRID_LOST_FRAMES` (consecutive frames with no ArUco
  marker at all before the mat counts as removed and the artifact ids are reset).

---

## Artifact tracking

Every detected fragment is tracked across inference cycles (`tracker.py`) so it
keeps a **stable identity, pose and color** once placed, instead of being
re-estimated (and re-colored) every time BiRefNet runs or the ArUco grid is
re-detected:

- A new detection must be seen for `confirm_hits` (default 3) **consecutive**
  cycles before it becomes an artifact — BiRefNet sometimes hallucinates objects
  out of the empty mat's texture for a cycle or two, and those phantoms never
  survive the confirmation. At confirmation the blob must also rise at least
  `min_artifact_height_mm` above its local surroundings (averaged over the
  confirmation cycles), or it is discarded as flat mat texture; a blob whose
  depth cannot be judged (dark material, no IR return) is always kept.
- A confirmed fragment gets an id and its pose is estimated **once**. Fragments
  confirmed in the same cycle (the normal case when the mat is placed with all
  its fragments already on it) get their ids in **reading order** of the grid:
  top row first, then left to right — id 1 is the top-left fragment. A fragment
  whose detection flickered during the confirmation window confirms later and
  simply takes the next id.
- On later cycles it is matched to its existing track by position; its pose is
  **not** recomputed as long as it keeps being matched. One exception: when the
  **grid really moves** (the ArUco node sends a new crop, i.e. a grid corner
  moved more than `CROP_CHANGE_PX`), the fragments moved with the mat, so every
  stored pose is dropped and re-estimated at the new position — the ids and
  colors are kept. Grid re-detection jitter below that threshold still never
  touches the poses.
- If a fragment is not detected for more than `miss_limit` consecutive cycles,
  its track is removed. If a detection then reappears **at the same spot** within
  `reacquire_window_s` seconds (a fragment whose segmentation flickered on and
  off), the old id and pose are **reacquired** instead of assigning a new id —
  the reacquisition happens when the reappeared detection passes the same
  `confirm_hits` confirmation, so a one-cycle phantom can never resurrect an old
  id. A fragment that reappears somewhere else still gets a new id.
- While the scene is **occluded** (a hand or robot arm over the table, detected
  from the depth image — see `DepthFilterConfig` above), miss counting is frozen:
  covered fragments never expire during a pick/place, and keep their id, pose,
  color and grid cells when the hand leaves. Blobs that stick out of the table
  more than `max_artifact_height_mm` are never counted as fragments, so the hand
  itself does not occupy grid cells or get published over MQTT.
- Physically moving a fragment is handled as remove-then-add: the old id expires
  after `miss_limit` cycles while the new position appears immediately under a
  new id, so both can briefly coexist in the published list during a move.
- When **no ArUco marker is visible** for `GRID_LOST_FRAMES` consecutive frames
  (the whole mat was taken off the table), the tracker is **reset**: every id is
  forgotten (an empty artifact list is published) and the next batch of
  fragments starts again from id 1. The reset only fires once the table is
  really empty and not occluded, so a hand or the robot arm briefly covering
  all the markers can never wipe the ids of fragments still on the mat. Leave
  the table clear for a few seconds between one mat and the next, or the old
  ids may survive into the new batch.

### Daily examination records

The ArUco node keeps a human-readable daily log in **`artefact_records.txt`**
(written in the working directory, next to `result.png`). Every examination
session — a mat placed under the camera with its artefacts — becomes one block
with the session number, the start time, the artefact count and one line per
artefact (id, occupied cells, grasp points). A summary with the day's totals
stays at the top of the file:

```
ARTEFACT RECORDS - daily log
Examinations today: 2 | Artefacts examined today: 7
==================================================

--------------------------------------------------
Examination #1 - 2026-07-17 09:12:44
artefacts: 4 | coordinates in mm
--------------------------------------------------
id 1 | cells: A2 A3 | grasp 1: 100, -50, 700 | grasp 2: 110, -40, 702
...
```

The current session appears in the file too, marked `(in progress)`, and is
finalized when the mat is removed — so the last session of the day is saved
even if the app is stopped with the mat still on the table. The file survives
app restarts (the counters are reloaded from it); it only restarts from zero
with the **Clear daily log** button on the web page.

### Published message format

The inference stage publishes the full list of currently tracked artifacts (that
have a valid pose) to `--mqtt-send-topic` as a JSON string, **only when the list
actually changes** (an artifact was added, removed, or just got its first valid
pose) plus a periodic heartbeat (`TrackerConfig.heartbeat_s`, default every 5s)
so a consumer that missed a message stays in sync:

```json
[
  {
    "object_number": 1,
    "x_1": 0.123, "y_1": -0.045, "z_1": 0.512,
    "x_2": 0.130, "y_2": -0.060, "z_2": 0.515
  }
]
```

`object_number` is the artifact's **persistent tracking id** (not a per-frame
index). Coordinates are in the camera's physical units (meters for RealSense,
millimeters for Orbbec, following each SDK's deprojection). An artifact whose
depth reads as `0` at either grasp-segment endpoint is kept as a track but left
out of the published list until a valid reading is obtained. When the last
artifact is removed, an **empty list `[]` is published** — this is how a
consumer knows the table was cleared.

---

## MQTT configuration

### Connection options

The broker is configured with the `--mqtt-*` flags listed above
(`--mqtt-host`, `--mqtt-port`, `--mqtt-user`, `--mqtt-password`,
`--mqtt-send-topic`). The client connects on startup and publishes continuously.
See [Artifact tracking](#artifact-tracking) above for when a message is actually
sent and what it contains.

### Bundled Mosquitto broker

`docker-compose.yml` (and the Orbbec variant) start an `eclipse-mosquitto`
broker alongside the app, configured by `mqtt/mosquitto-config/mosquitto.conf`:

- `1883` — plain MQTT listener
- `8080` — MQTT over WebSockets
- `9001` — additional mapped port
- authentication via `mqtt/auth/pwd.txt`

Two users are provisioned in `pwd.txt` (`mqtt` and `pippo`). To add or change a
user with the Mosquitto tooling:

```bash
# Add/update a user (will prompt for the password)
docker run --rm -it -v "$(pwd)/mqtt/auth:/auth" eclipse-mosquitto \
    mosquitto_passwd /auth/pwd.txt <username>
```

If you point the app at your own broker instead, simply set the `--mqtt-*`
flags and you can ignore the bundled service.

---

## Running with Docker

The Compose files bring up both the Mosquitto broker and the camera
application. They run the container with `privileged: true`, `network_mode:
host`, and mount `/dev` and the X11 socket so the camera (USB) and the on-screen
output work.

### RealSense

```bash
docker compose build
docker compose up
```

The RealSense image builds `librealsense` (2.56.3) from source with Python
bindings. The container installs the package and dependencies, then runs
`object_extraction.py` with default (RealSense) settings.

### Orbbec

```bash
docker compose -f docker-compose.orbbec.yml build
docker compose -f docker-compose.orbbec.yml up
```

The Orbbec image is based on Ubuntu, installs the dependencies (including the
Orbbec SDK) in a virtualenv, and runs the app with
`--camera-type orbbec --stream-width 1280 --stream-height 720`. This Compose
file also reserves one NVIDIA GPU for the container.

> Before launching, allow the container to use your X display if you want to
> see windows/output locally:
> ```bash
> xhost +local:
> ```

---

## Output

- **Web page at `http://localhost:8090`** — two live videos side by side (the
  ArUco grid view and the annotated inference view) plus a realtime panel with
  each artifact's id, occupied cells and grasp pose (endpoints in the camera
  frame, m for RealSense / mm for Orbbec). The individual streams are also
  available at `/aruco` and `/inference`, the raw info JSON at `/data`.
- **`result.png`** — annotated inference frame, overwritten every iteration.
  Shows the detected contours, centroids, and the grasp segment with its
  endpoints, each artifact drawn in its stable color with its id.
- **`aruco_result.png`** — the grid view snapshot, written every few seconds.
- **MQTT messages** — published to `--mqtt-send-topic` as described above.

---

## Troubleshooting

- **`ModuleNotFoundError: BiRefNet` / `config`** — the submodule was not cloned.
  Run `git submodule update --init --recursive`.
- **`Depth camera with Color sensor required` (RealSense)** — the camera was not
  detected; check the USB connection and permissions.
- **Camera stalls / USB overflow (RealSense)** — the driver detects stale frames
  and automatically restarts the pipeline; this is expected and logged.
- **`pyorbbecsdk is required ...`** — install the Orbbec SDK, or run in
  RealSense mode.
- **Slow inference** — confirm CUDA is available; on CPU the model runs but is
  significantly slower.

---

## Contact

For questions about this project, contact `giuliano.dami@qbrobotics.com`.
