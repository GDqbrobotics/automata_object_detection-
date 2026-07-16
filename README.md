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
| `crop_width` / `crop_height` | `500` / `300` | Size of the region of interest that is segmented. |
| `crop_starting_row` / `crop_starting_col` | centered in 1280×720 | Top-left corner of the crop. |

Only the area inside this crop is processed, which keeps inference fast and
limits detection to the working surface. Adjust these values to move or resize
the region of interest.

Other tunables that currently live in the code (not CLI flags):

- **Contour size filter** in `inference.py`: contours with fewer than `200` or
  more than `1500` points are ignored — this rejects noise and oversized blobs.
- **Alpha threshold** for turning the segmentation mask into a binary image
  (`254` in `inference.py`).
- **Depth range** for Orbbec in `camera_orbbec.py`: `MIN_DEPTH = 20 mm`,
  `MAX_DEPTH = 10000 mm`.
- **BiRefNet input size**: `1024×1024` in `model.py`.

---

## MQTT configuration

### Connection options

The broker is configured with the `--mqtt-*` flags listed above
(`--mqtt-host`, `--mqtt-port`, `--mqtt-user`, `--mqtt-password`,
`--mqtt-send-topic`). The client connects on startup and publishes continuously.

### Published message format

For each processed frame the inference stage builds a list with one entry per
detected object and publishes it to `--mqtt-send-topic` as a JSON string.
Each object contains the two endpoints of its grasp segment in 3D:

```json
[
  {
    "object_number": 0,
    "x_1": 0.123, "y_1": -0.045, "z_1": 0.512,
    "x_2": 0.130, "y_2": -0.060, "z_2": 0.515
  }
]
```

Coordinates are in the camera's physical units (meters for RealSense,
millimeters for Orbbec, following each SDK's deprojection). Objects whose depth
reads as `0` at either endpoint (invalid/missing depth) are skipped, and frames
with no valid object produce no message.

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

- **`result.png`** — annotated frame, overwritten every iteration. Shows the
  detected contours (red), centroids (green dots), and the grasp segment with
  its endpoints (green line, yellow endpoints), plus a Unix timestamp.
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
