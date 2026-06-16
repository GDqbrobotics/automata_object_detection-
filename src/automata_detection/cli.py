import argparse
from multiprocessing import Process, Queue

from .inference import start_inference
from .mqtt import mqtt_send


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automata object detection with depth camera and MQTT")

    parser.add_argument("--camera-type", type=str, default="realsense", choices=["realsense", "orbbec"], help="Type of depth camera to use")
    parser.add_argument("--stream-width", type=int, default=1920, help="Width of the RGB stream")
    parser.add_argument("--stream-height", type=int, default=1080, help="Height of the RGB stream")
    parser.add_argument("--mqtt-host", type=str, default="192.168.139.70", help="MQTT broker host")
    parser.add_argument("--mqtt-port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--mqtt-user", type=str, default="mqtt", help="MQTT username")
    parser.add_argument("--mqtt-password", type=str, default="Vn370gi@lo#T", help="MQTT password")
    parser.add_argument("--mqtt-send-topic", type=str, default="test_coordinate", help="MQTT topic for events")
    parser.add_argument("--inference-sleep", type=float, default=0.01, help="Sleep time between inference iterations")
    parser.add_argument("--verbose", action="store_true", help="Print detailed debug information")

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    # Import the appropriate camera module based on camera type
    if args.camera_type == "realsense":
        from .camera import read_camera
        camera_name = "RealSense"
    elif args.camera_type == "orbbec":
        from .camera_orbbec import read_camera
        camera_name = "Orbbec"
    else:
        raise ValueError(f"Unknown camera type: {args.camera_type}")

    if args.verbose:
        print(f"[CLI] Using {camera_name} camera")

    frame_queue = Queue(maxsize=1)
    parameters_queue = Queue(maxsize=1)
    send_queue = Queue()

    read_process = Process(
        target=read_camera,
        kwargs={
            "width": args.stream_width,
            "height": args.stream_height,
            "frame_queue": frame_queue,
            "parameters_queue": parameters_queue,
            "verbose": args.verbose,
        },
    )

    send_process = Process(
        target=mqtt_send,
        kwargs={
            "topic": args.mqtt_send_topic,
            "host": args.mqtt_host,
            "port": args.mqtt_port,
            "send_queue": send_queue,
            "username": args.mqtt_user,
            "password": args.mqtt_password,
            "verbose": args.verbose,
        },
    )

    inference_process = Process(
        target=start_inference,
        kwargs={
            "frame_queue": frame_queue,
            "parameters_queue": parameters_queue,
            "send_queue": send_queue,
            "verbose": args.verbose,
            "sleep": args.inference_sleep,
            "camera_type": args.camera_type,
        },
    )

    read_process.start()
    send_process.start()
    inference_process.start()
    inference_process.join()
