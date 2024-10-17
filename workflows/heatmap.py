# This script leverages multiple models and features from Ultralytics, licensed under AGPL-3.0.
# See: https://github.com/ultralytics/ for more information.

import argparse
import cv2
from ultralytics import YOLO
from ultralytics.solutions import heatmap

parser = argparse.ArgumentParser(description="Heatmap line counting")
parser.add_argument("-v", "--video_path", help="Video path")
arguments = parser.parse_args()
video_path = arguments.video_path


model = YOLO("yolov8x.pt")
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"

# Video writer
video_writer = cv2.VideoWriter(
    "heatmap_output.avi",
    cv2.VideoWriter_fourcc(*"mp4v"),
    int(cap.get(5)),
    (int(cap.get(3)), int(cap.get(4))),
)

classes_for_heatmap = [0, 2]  # classes for heatmap (Optional)

# Init heatmap
heatmap_obj = heatmap.Heatmap()
# decay_factor: Used for removing heatmap after an object is no longer
# in the frame, its value should also be in the range (0.0 - 1.0).
heatmap_obj.set_args(
    colormap=cv2.COLORMAP_PARULA,
    imw=cap.get(3),  # should same as cap height
    imh=cap.get(4),  # should same as cap width
    view_img=True,
    shape="circle",
    decay_factor=0.99,
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print(
            "Video frame is empty or video processing has been successfully completed."
        )
        break
    tracks = model.track(
        im0,
        persist=True,
        show=True,
        # classes=classes_for_heatmap
    )

    im0 = heatmap_obj.generate_heatmap(im0, tracks)
    video_writer.write(im0)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()
