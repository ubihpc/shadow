# This script leverages multiple models and features from Ultralytics, licensed under AGPL-3.0.
# See: https://github.com/ultralytics/ for more information.

import argparse
import cv2
import os
import subprocess
from ultralytics import YOLO


def predict(chosen_model, img, classes=[0], conf=0.3):
    if classes:
        results = chosen_model.predict(img, classes=[0], conf=conf, imgsz=800 * 8)
    else:
        results = chosen_model.predict(img, conf=conf)
    return results


def predict_and_detect(chosen_model, img, classes=[0], conf=0.3):
    results = predict(chosen_model, img, classes, conf=conf)
    return results


def pixelate_region(roi, pixel_size=10):
    # Resize input to a smaller size
    height, width = roi.shape[:2]
    temp = cv2.resize(roi, (pixel_size, pixel_size), interpolation=cv2.INTER_LINEAR)
    # Resize back to original size
    return cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)


def anonymize_detections(img, results):
    for result in results:
        for box in result.boxes:
            cls = box.cls
            cls_int = int(cls.item())

            if cls_int != 0:  # Filter out non-person detections
                continue

            x1, y1, x2, y2 = (
                int(box.xyxy[0][0]),
                int(box.xyxy[0][1]),
                int(box.xyxy[0][2]),
                int(box.xyxy[0][3]),
            )

            img_width = img.shape[1]
            right_boundary = 0.8 * img_width
            left_boundary = 0.2 * img_width

            # Do not pixelate people in the right corner of the frame (x > 0.8 * img.shape[1])
            if x1 > right_boundary or x1 < left_boundary:
                continue

            # Only pixelate the face region (0.2 * height) of the person
            y2 = y1 + int(0.3 * (y2 - y1))

            roi = img[y1:y2, x1:x2]
            pixelated_roi = pixelate_region(roi, pixel_size=15)
            img[y1:y2, x1:x2] = pixelated_roi
    return img


parser = argparse.ArgumentParser(description="Faces anonymization")
parser.add_argument("-v", "--video_path", help="Video path")
arguments = parser.parse_args()

# Paths
video_path = arguments.video_path
output_video_path = os.path.join(os.path.dirname(video_path), "processed_video.mp4")
final_output_path = os.path.join(os.path.dirname(video_path), "final_output.mp4")

model = YOLO("yolo11x.pt")
cap = cv2.VideoCapture(video_path)

# Start video at frame 15
# cap.set(cv2.CAP_PROP_POS_FRAMES, 15*34)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
out = cv2.VideoWriter(
    output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height)
)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = predict_and_detect(model, frame, classes=[], conf=0.1)
    result_frame = anonymize_detections(frame, results)

    # Display the frame
    # cv2.namedWindow("Anonymized", cv2.WINDOW_GUI_NORMAL)
    # cv2.imshow("Anonymized", result_frame)

    cv2.imwrite("anon.jpg", result_frame)
    out.write(result_frame)

    # Break the loop if 'q' is pressed
    # if cv2.waitKey(1) & 0xFF == ord("q"):
    #    break

cap.release()
out.release()
cv2.destroyAllWindows()

# Use ffmpeg to combine the original audio with the processed video
subprocess.run(
    [
        "ffmpeg",
        "-i",
        output_video_path,
        "-i",
        video_path,
        "-c",
        "copy",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        final_output_path,
    ]
)
