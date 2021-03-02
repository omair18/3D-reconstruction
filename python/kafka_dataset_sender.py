import os
import cv2
import argparse
import time
import uuid
import json
from kafka import KafkaProducer


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i',
        '--input',
        type=str,
        help='Path to a folder with images or to a video',
        required=True
    )

    parser.add_argument(
        '-fs',
        '--frame-step',
        type=int,
        help='Frame step.',
        required=True
    )

    parser.add_argument(
        '-b',
        '--broker',
        type=str,
        help='URL of Kafka broker.',
        required=True
    )

    parser.add_argument(
        '-ci',
        '--camera-id',
        type=int,
        help='Camera id',
        required=True
    )

    parser.add_argument(
        '-jq',
        '--jpeg-quality',
        type=int,
        help='JPEG quality',
        required=False,
        default=40
    )

    parser.add_argument(
        '-t',
        '--topic',
        type=str,
        help='Topic of Kafka broker.',
        required=True
    )

    parser.add_argument(
        '-gi',
        '--group-id',
        type=int,
        help='Group id of Kafka broker.',
        required=True
    )

    args = parser.parse_args()

    input_path = args.input
    frame_step = args.frame_step
    jpeg_quality = args.jpeg_quality
    camera_id = args.camera_id
    broker = args.broker
    topic = args.topic
    group_id = args.group_id

    frames = []
    dataset_uuid = str(uuid.uuid4())

    if not os.path.exists(input_path):
        raise RuntimeError("Invalid input path!")

    if frame_step <= 0:
        raise RuntimeError("Frame step must be positive!")

    print("Reading input ...")

    if os.path.isfile(input_path):
        capture = cv2.VideoCapture(input_path)
        if not capture.isOpened():
            raise RuntimeError("Failed to read input file.")
        frames_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(0, frames_total, frame_step):
            capture.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, image = capture.read()
            if image is not None:
                frames.append(image)
            else:
                print("Failed to read frame with id %d", i)
    elif os.path.isdir(input_path):
        files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]
        files_count = len(files)
        for i in range(0, files_count, frame_step):
            image_path = os.path.join(input_path, files[i])
            image = cv2.imread(image_path)
            if image is not None:
                frames.append(image)
            else:
                print("Failed to read image %s", image_path)
    else:
        raise RuntimeError("Input path must be a path to either a video or a folder with images!")

    print("Encoding images ...")
    encoded_images = []
    images_total = len(frames)
    for i in range(0, images_total):
        success, buffer = cv2.imencode('.jpg', frames[i], [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        if not success:
            print("Failed to encode image with id = %d", i)
        else:
            encoded_images.append(buffer)

    print("Sending images ...")
    producer = KafkaProducer(bootstrap_servers=broker)

    encoded_images_total = len(encoded_images)
    for i in range(0, encoded_images_total):
        key_json_str = {
            'cameraID': camera_id,
            'timestamp': int(round(time.time() * 1000)),
            'UUID': dataset_uuid,
            'frameId': i,
            'framesTotal': encoded_images_total
        }
        key_json = json.dumps(key_json_str)
        print(key_json)
        producer.send(topic, key=str(key_json).encode('utf-8'), value=encoded_images[i].tobytes())


if __name__ == '__main__':
    main()
