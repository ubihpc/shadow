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

line_points = [
    (852, 198),
    (1357, 807),
]  # [(345, 427), (694, 532)]  # line for object counting

# Init heatmap
heatmap_obj = heatmap.Heatmap()
heatmap_obj.set_args(
    colormap=cv2.COLORMAP_PARULA,
    imw=cap.get(4),  # should same as cap height
    imh=cap.get(3),  # should same as cap width
    view_img=True,
    shape="circle",
    count_reg_pts=line_points,
    count_txt_color=(255, 255, 255),
    count_color=(0, 0, 0),
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print(
            "Video frame is empty or video processing has been successfully completed."
        )
        break
    tracks = model.track(im0, persist=True, show=True)

    im0 = heatmap_obj.generate_heatmap(im0, tracks)
    video_writer.write(im0)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()
