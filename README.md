# Automata Object Detection

Automata Object Detection is a vision system for identifying archaeological fragments such as ceramics and lithics using a depth camera (RealSense or Orbbec) and the BiRefNet segmentation model.

## Project structure

- `object_extraction.py` - thin application entrypoint for compatibility.
- `src/automata_detection/` - Python package with camera, model, inference, pose, MQTT, and CLI modules.
  - `camera.py` - RealSense D415 camera driver
  - `camera_orbbec.py` - Orbbec depth camera driver (e.g., Astra2)
- `BiRefNet/` - model submodule used for segmentation.
- `requirements.txt` - Python dependencies for local development.
- `pyproject.toml` - package metadata and build config.
- `Dockerfile` / `Dockerfile.orbbec` - container definitions for RealSense and Orbbec.
- `docker-compose.yml` / `docker-compose.orbbec.yml` - orchestration files.

## Installation

1. Clone the repository with submodules:
   ```bash
   git clone --recurse-submodules https://github.com/GDqbrobotics/automata_object_detection-.git
   cd automata_detection
   ```

2. Install Python dependencies:
   ```bash
   python3 -m pip install -r requirements.txt
   ```

3. If you prefer containerized execution, build the appropriate Docker image:
   ```bash
   # For RealSense:
   docker compose build
   
   # For Orbbec:
   docker compose -f docker-compose.orbbec.yml build
   ```

## Usage

### Local execution

Run the main application with RealSense (default):
```bash
python3 object_extraction.py --verbose
```

Run with Orbbec camera:
```bash
python3 object_extraction.py --camera-type orbbec --verbose
```

### Docker execution

Start with RealSense:
```bash
docker compose up
```

Start with Orbbec:
```bash
docker compose -f docker-compose.orbbec.yml up
```

The application captures RGB and depth frames, segments objects, computes pose coordinates, publishes MQTT events, and writes `result.png`.

### MQTT Topics

The system uses three MQTT topics for state management and control:

| Topic | Direction | Payload | Description |
|-------|-----------|---------|-------------|
| `/status` | Publish | `{"status": "INITIALIZING\|READY\|DETECTING\|CAPTURING"}` | System state updates |
| `/results` | Publish | `{"type":"object_points","object_id":"object_004","points":[[x,y,z],[x,y,z]]}` | One message per object with averaged poses (meters) |
| `/command` | Subscribe | `{"event": "capture", "object_id": optional}` | Trigger capture phase |

**State Machine:**
- `INITIALIZING` - System starting up, loading model, connecting to camera
- `READY` - Model loaded, camera running, waiting for capture command
- `DETECTING` - Running one-frame detection to list all objects (general capture only)
- `CAPTURING` - Collecting N frames (default 10), computing averaged poses, saving cropped object images to `captures/`

**Command Examples:**
```bash
# Capture all detected objects (10 frames)
mosquitto_pub -t /command -m '{"event": "capture"}'

# Capture only object with ID 0
mosquitto_pub -t /command -m '{"event": "capture", "object_id": 0}'
```

**Result Message Format** (published to `/results`, one per object):
```json
{
  "type": "object_points",
  "object_id": "object_004",
  "points": [[0.104, 0.212, 0.054], [0.121, 0.223, 0.061]]
}
```
- Coordinates in **meters** (converted from mm)
- `object_id`: zero-padded 3-digit format (object_000, object_001, ...)
- `points`: array of two 3D points [x, y, z] representing the object segment

**Objects Detected Message** (published to `/results` when general capture is requested):
```json
{
  "type": "objects_detected",
  "objects": ["object_000", "object_001", "object_002", "object_003", "object_004"],
  "count": 5
}
```
- Published immediately after a general capture command (no `object_id`)
- Lists all detected objects in the scene
- `count`: total number of detected objects

### Command-line options

```
--camera-type {realsense,orbbec}
                      Type of depth camera to use (default: realsense)
--stream-width WIDTH  Width of the RGB stream (default: 1920)
--stream-height HEIGHT
                      Height of the RGB stream (default: 1080)
--mqtt-host HOST      MQTT broker host (default: 192.168.139.70)
--mqtt-port PORT      MQTT broker port (default: 1883)
--mqtt-user USER      MQTT username (default: mqtt)
--mqtt-password PASS  MQTT password (default: Vn370gi@lo#T)
--mqtt-status-topic TOPIC
                      MQTT topic for status updates (default: /status)
--mqtt-results-topic TOPIC
                      MQTT topic for capture results (default: /results)
--mqtt-command-topic TOPIC
                      MQTT topic for commands (default: /command)
--capture-frames N    Number of frames to capture for averaging (default: 10)
--inference-sleep SEC Sleep time between inference iterations (default: 0.01)
--verbose             Print detailed debug information
```

### Captured Object Images

At the end of each CAPTURING phase, the system saves cropped images of each detected object to the `captures/` directory:

- **Directory**: `captures/` (created automatically)
- **Naming**: `object_<ID>_<timestamp>.jpg` (e.g., `object_0_1703123456789.jpg`)
- **Content**: Cropped region around the object's contour with 10px padding
- **Selection**: Uses the middle frame of the capture sequence for best representation

If `object_id` is specified in the capture command, only that object's image is saved. Otherwise, images for all detected objects are saved.

## Development

The reorganized package is designed for maintainability:

- `src/automata_detection/camera.py` - RealSense D415 capture, temporal filtering, and cropping.
- `src/automata_detection/camera_orbbec.py` - Orbbec depth camera capture with similar API.
- `src/automata_detection/model.py` - BiRefNet loading and segmentation preprocessing.
- `src/automata_detection/inference.py` - object extraction, contour analysis, result generation with MQTT state machine, and cropped object image capture.
- `src/automata_detection/pose.py` - coordinate conversion (supports both camera types) and pose message construction.
- `src/automata_detection/mqtt.py` - MQTT publishing and subscription (both process-based and threaded clients).
- `src/automata_detection/cli.py` - command-line interface and process orchestration.

## Camera-specific notes

### RealSense D415
- Requires `pyrealsense2` package
- Uses Intel's RealSense SDK for frame capture and alignment
- Supports RGB and depth streams at 15 fps

### Orbbec Astra2 / Gemini
- Requires `pyorbbecsdk` package
- Uses Orbbec's OpenBody SDK for frame capture
- Supports RGB and depth streams at 15 fps
- Coordinate conversion using camera intrinsics

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Make your changes.
4. Open a pull request.

## Contact

For questions, contact `giuliano.dami@qbrobotics.com`.