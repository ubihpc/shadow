# This script leverages multiple models and features from Ultralytics, licensed under AGPL-3.0.
# See: https://github.com/ultralytics/ for more information.


import argparse
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.solutions import heatmap


parser = argparse.ArgumentParser(description="Heatmap region counting")
parser.add_argument("-v", "--video_path", help="Video path")
arguments = parser.parse_args()
video_path = arguments.video_path

# model = YOLO("yolov8n.pt")
model = YOLO("yolov8x.pt")
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"

# Video writer
video_writer = cv2.VideoWriter(
    "heatmap_region_output.avi",
    cv2.VideoWriter_fourcc(*"mp4v"),
    int(cap.get(5)),
    (int(cap.get(3)), int(cap.get(4))),
)

# Define region points
region_points = [(20, 400), (1080, 404), (1080, 360), (20, 360)]


# Mouse callback function to handle events
def draw_polygon(event, x, y, flags, param):
    global polygon_points, region_points

    if event == cv2.EVENT_LBUTTONDOWN:
        # Add the clicked point to the polygon points list
        polygon_points.append((x, y))

        # Draw a circle at the clicked point
        cv2.circle(im0, (x, y), 5, (0, 0, 255), -1)

        # Draw lines between consecutive points to form the polygon
        if len(polygon_points) > 1:
            cv2.line(im0, polygon_points[-2], polygon_points[-1], (0, 0, 255), 2)

        # Draw a line between the last and first points to close the polygon
        if len(polygon_points) > 2:
            cv2.line(im0, polygon_points[-1], polygon_points[0], (0, 0, 255), 2)
        region_points = polygon_points


# Init heatmap
heatmap_obj = heatmap.Heatmap()
heatmap_obj.set_args(
    colormap=cv2.COLORMAP_PARULA,
    imw=cap.get(4),  # should same as cap height
    imh=cap.get(3),  # should same as cap width
    view_img=True,
    shape="circle",
    count_reg_pts=region_points,
)

polygon_points = []
success, im0 = cap.read()

# Create a window and set the mouse callback function
cv2.namedWindow("Draw Polygon")
cv2.setMouseCallback("Draw Polygon", draw_polygon)

# Display the image and wait for the user to draw the polygon
while True:
    cv2.imshow("Draw Polygon", im0)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Convert the polygon points to numpy array
polygon_points = np.array(polygon_points)

# Draw the polygon on the image
cv2.fillPoly(im0, [polygon_points], (0, 255, 0))
heatmap_obj.set_args(
    colormap=cv2.COLORMAP_PARULA,
    imw=cap.get(3),  # should same as cap height
    imh=cap.get(4),  # should same as cap width
    view_img=True,
    shape="circle",
    count_reg_pts=region_points,
)

while cap.isOpened():
    success, im0 = cap.read()
    # Draw a rectangle on the image to define the region with mouse clicks
    # Create a list to store the polygon points

    if not success:
        print(
            "Video frame is empty or video processing has been successfully completed."
        )
        break
    tracks = model.track(im0, persist=True, show=False)

    im0 = heatmap_obj.generate_heatmap(im0, tracks)
    video_writer.write(im0)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()
