# Imports
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

import time
import cv2
import numpy as np
import argparse 
import json
#import paho.mqtt.client as mqtt
from multiprocessing import Process, Queue
from PIL import ImageOps
import sys
sys.path.insert(0, '/home/workspace/BiRefNet')

from BiRefNet.models.birefnet import BiRefNet

##########################
#### RealSense Utils :####
##########################
import pyrealsense2 as rs

cameraInfo = dict(
    height=720, #depth height
    width=1280, #depth width
    K=[890.5523071289062, 0.0, 639.445068359375, 0.0, 890.5523071289062, 363.5865783691406, 0.0, 0.0, 1.0],
    D=[0.0, 0.0, 0.0, 0.0, 0.0]
)

CROP_WIDTH = 1280   
CROP_HEIGHT = 720
CROP_STARTING_ROW = 0
CROP_STARTING_COL = 320

class TemporalFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.previous_frame = None

    def process(self, frame):
        if self.previous_frame is None:
            result = frame
        else:
            result = cv2.addWeighted(frame, self.alpha, self.previous_frame, 1 - self.alpha, 0)
        self.previous_frame = result
        return result

class ImageCropper:
    def __init__(self, width, height, starting_row, starting_col):
        self.width = width
        self.height = height
        self.starting_row = starting_row
        self.starting_col = starting_col

    def crop(self, image):
        end_row, end_col = self.starting_row + self.height, self.starting_col + self.width
        return image.crop((self.starting_col, self.starting_row, end_col, end_row))
    
    def cropped2orig(self, row, col):
        return row + self.starting_row, col + self.starting_col

def convert_depth_to_phys_coord_using_realsense(x, y, depth):
    _intrinsics = rs.intrinsics()
    _intrinsics.width = cameraInfo["width"]
    _intrinsics.height = cameraInfo["height"]
    _intrinsics.ppx = cameraInfo["K"][2]
    _intrinsics.ppy = cameraInfo["K"][5]
    _intrinsics.fx = cameraInfo["K"][0]
    _intrinsics.fy = cameraInfo["K"][4]
    _intrinsics.model = rs.distortion.none
    _intrinsics.coeffs = cameraInfo["D"]
    result = rs.rs2_deproject_pixel_to_point(_intrinsics, [x, y], depth)
    return result[2], -result[0], -result[1]
#############################
#### End RealSense Utils ####
#############################

def read_camera(*, frame_queue, width, height, verbose=False):
    pipeline = rs.pipeline()

    config = rs.config()

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("Depth camera with Color sensor required")
        exit(0)
    
    config.enable_stream(rs.stream.depth, cameraInfo["width"], cameraInfo["height"], rs.format.z16, 15)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 15)

    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)
    
    s = profile.get_device().query_sensors()[1]
    s.set_option(rs.option.saturation, 70)
    s.set_option(rs.option.contrast, 65)
    s.set_option(rs.option.exposure, 45)

    clipping_distance_in_meters = 1
    clipping_distance = clipping_distance_in_meters / depth_scale

    align_to = rs.stream.color
    align = rs.align(align_to)

    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        if verbose: print("[STREAM] Read rgb frame of size", color_image.shape)
        if verbose: print("[STREAM] Read depth frame of size", depth_image.shape)
        
        # Apply temporal filtering
        temporal_filter = TemporalFilter(alpha=0.5)
        depth_image = temporal_filter.process(depth_image)
        
        if not frame_queue.full():
            frame_queue.put((color_image, depth_image))


def extract_object(birefnet, image):
    # Data settings
    image_size = (1024, 1024)
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # image = Image.open(imagepath)
    input_images = transform_image(image).unsqueeze(0).to('cuda').half()

    # Prediction
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size)
    image.putalpha(mask)
    return image, mask

def inference(*, frame_queue, verbose=False, sleep=0, depth_height=720, depth_width=1280):
    
    # Load weights from Hugging Face Models
    birefnet = BiRefNet.from_pretrained('ZhengPeng7/BiRefNet')
    torch.set_float32_matmul_precision(['high', 'highest'][0])
    birefnet.to('cuda')
    birefnet.eval()
    birefnet.half()
    print("[INFERENCE] Loaded model")

    while True:
        frame, depth = frame_queue.get()
        # print("[INFERENCE] Inference on frame of size", frame.shape)
        # print("[INFERENCE] Inference on depth of size", depth.shape)
        coeff_height = depth_height / frame.shape[0]
        coeff_width = depth_width / frame.shape[1]

        base_image = Image.fromarray(frame)

        cropper = ImageCropper(CROP_WIDTH, CROP_HEIGHT, CROP_STARTING_ROW, CROP_STARTING_COL)
        base_image = cropper.crop(base_image)
        
        if verbose: print("[INFERENCE] Inference on image of size", base_image.size)

        # Visualization
        cv2.imwrite('debug_bkgrnd.png', frame)

        base_image1 = np.array(base_image.convert('RGBA'))
        base_image1 = cv2.cvtColor(base_image1, cv2.COLOR_RGBA2BGRA)

        # Step 1: Load and preprocess the image
        # image = Image.open('ceramic_out.png')
        image = extract_object(birefnet, base_image)[0]
        image = np.array(image.convert('RGBA'))
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)


        # Step 2: Thresholding to separate the object from the background
        _, thresholded = cv2.threshold(image[:, :, 3], 254, 255, cv2.THRESH_BINARY_INV)

        cv2.imwrite('debug.png', image)
        contours, hierarchy = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) #CHAIN_APPROX_SIMPLE

        results = []

        canvas = np.ones_like(image, dtype=np.uint8) * 255  # bianco

        for contour in contours:
            if contour.size < 500 or contour.size > 3000:
                continue
            M = cv2.moments(contour)
            m00 = M['m00']
            if m00 != 0:
                cx = int(M['m10']/m00)
                cy = int(M['m01']/m00)

                cv2.drawContours(base_image1, [contour], -1, (255, 0, 0, 255), 2)   # rosso
                cv2.circle(base_image1, (cx, cy), 6, (0, 255, 0, 255), 2)          # verde


                segment_p1, segment_p2 = findMinSegment(cx, cy, contour)

                cv2.line(base_image1, segment_p1, segment_p2, (0, 255, 0), 2)
                cv2.circle(base_image1, segment_p1, 7, (0, 255, 255), -1)
                cv2.circle(base_image1, segment_p2, 7, (0, 255, 255), -1)

                result = {'1': segment_p1, '2': segment_p2}
                results.append(result)

        cv2.imwrite("result.png", base_image1)

        if sleep > 0:
            time.sleep(sleep)

def findMinSegment(cx, cy, contour):

    contour = contour.squeeze(axis=1) # converto contorno da (N, 1, 2) a (N, 2) per facilitare i calcoli

    # 1. Traduci i punti del contorno in modo che il centroide sia all'origine (0,0).
    translated_contour = contour - [cx, cy]

    # 2. Calcola l'angolo di ogni punto rispetto al centroide.
    # np.arctan2 restituisce l'angolo in radianti nell'intervallo [-pi, pi].
    angles = np.arctan2(translated_contour[:, 1], translated_contour[:, 0])

    # 3. Ordina i punti del contorno in base al loro angolo.
    sorted_indices = np.argsort(angles)
    sorted_points = contour[sorted_indices]
    sorted_angles = angles[sorted_indices]

    min_dist_sq = float('inf')
    result_points = (None, None)

    # 4. Per ogni punto, trova il suo punto "opposto" più vicino usando la ricerca binaria.
    num_points = len(sorted_points)
    for i in range(num_points):
        p1 = sorted_points[i]
        angle1 = sorted_angles[i]

        # L'angolo opposto è sfasato di 180 gradi (pi radianti).
        target_angle = angle1 + np.pi
        # Normalizza l'angolo target per rimanere nell'intervallo [-pi, pi].
        if target_angle > np.pi:
            target_angle -= 2 * np.pi

        # 5. Usa np.searchsorted (ricerca binaria) per trovare l'indice del punto
        # con l'angolo più vicino a quello target.
        # Questo è molto più veloce di un ciclo annidato (O(log N) invece di O(N)).
        idx = np.searchsorted(sorted_angles, target_angle)

        # La ricerca binaria ci dà un punto di inserimento. Il punto opposto più vicino
        # sarà in questa posizione o in quella precedente. Controlliamo entrambe.
        # Usiamo il modulo (%) per gestire il "wrap-around" dell'array circolare di angoli.
        idx1 = idx % num_points
        idx2 = (idx - 1 + num_points) % num_points

        # Calcola la differenza angolare effettiva per entrambi i candidati.
        diff1 = np.abs(target_angle - sorted_angles[idx1])
        diff2 = np.abs(target_angle - sorted_angles[idx2])
        
        # Scegli il punto con la minima differenza angolare.
        best_idx = idx1 if diff1 < diff2 else idx2
        p2 = sorted_points[best_idx]

        # 6. Calcola la distanza e aggiorna il minimo se necessario.
        # Usare la distanza al quadrato è più veloce perché evita la radice quadrata.
        dist_sq = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            result_points = (tuple(p1), tuple(p2))

    return result_points

def pixel2pose(results,depth, coeff_height, coeff_width):    
    message = []
    cropper = ImageCropper(CROP_WIDTH, CROP_HEIGHT, CROP_STARTING_ROW, CROP_STARTING_COL)
    i=0
    for result in results:
        _y1,_x1 = cropper.cropped2orig(result['1'][1], result['1'][0])
        _x1 = _x1 * coeff_width
        _y1 = _y1 * coeff_height
        _z1 = depth[int(_y1), int(_x1)].item()

        z1, x1, y1 = convert_depth_to_phys_coord_using_realsense(_x1, _y1, _z1)

        _y2,_x2 = cropper.cropped2orig(result['2'][1], result['2'][0])
        _x2 = _x2 * coeff_width
        _y2 = _y2 * coeff_height
        _z2 = depth[int(_y2), int(_x2)].item()

        z2, x2, y2 = convert_depth_to_phys_coord_using_realsense(_x2, _y2, _z2)

        if z1*z2 != 0: #check for possible occlusion on the depth image
            message.append({
                "Object number": i,
                "x_1": x1,
                "y_1": y1,
                "z_1": z1,
                "x_2": x2,
                "y_2": y2,
                "z_2": z2
            })            
            i+=1

    return message

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stream-width",
        type=int,
        default=1920,
        help="Width of the stream"
    )

    parser.add_argument(
        "--stream-height",
        type=int,
        default=1080,
        help="Height of the stream"
    )

    # parser.add_argument(
    #     "--mqtt-host",
    #     type=str,
    #     default="192.168.139.70",
    #     help="Host of the MQTT broker"
    # )
    
    # parser.add_argument(
    #     "--mqtt-port",
    #     type=int,
    #     default=1883,
    #     help="Port of the MQTT broker"
    # )
    
    # parser.add_argument(
    #     "--mqtt-user",
    #     type=str,
    #     default="mqtt",
    #     help="MQTT username"
    # )

    # parser.add_argument(
    #     "--mqtt-password",
    #     type=str,
    #     default="Vn370gi@lo#T",
    #     help="MQTT password"
    # )

    # parser.add_argument(
    #     "--mqtt-send-topic",
    #     type=str,
    #     default="test_coordinate",
    #     help="MQTT topic to publish events to"
    # )

    # parser.add_argument(
    #     "--mqtt-send-topic-single",
    #     type=str,
    #     default="test_coordinate_single",
    #     help="MQTT topic to publish events to (single)"
    # )

    parser.add_argument(
        "--inference-sleep",
        type=float,
        default=0.01,
        help="Sleep time between inferences"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print more information"
    )

    args = parser.parse_args()

    frame_queue = Queue(maxsize=1)
    send_queue = Queue()

    read_process = Process(
        target=read_camera,
        kwargs=dict(
            width=args.stream_width,
            height=args.stream_height,
            frame_queue=frame_queue,
            verbose=args.verbose,
        )
    )

    # send_process = Process(
    #     target=send,
    #     kwargs=dict(
    #         topic=args.mqtt_send_topic,
    #         topic_single=args.mqtt_send_topic_single,
    #         host=args.mqtt_host,
    #         port=args.mqtt_port,
    #         send_queue=send_queue,
    #         username=args.mqtt_user,
    #         password=args.mqtt_password,
    #         verbose=args.verbose,
    #     )
    # )

    inference_process = Process(
        target=inference,
        kwargs=dict(
            frame_queue=frame_queue,
            verbose=args.verbose,
            sleep=args.inference_sleep,
        )
    )

    read_process.start()
    # send_process.start()
    inference_process.start()

    inference_process.join()
    # send_process.terminate()
    read_process.terminate()


if __name__ == "__main__":
    main()