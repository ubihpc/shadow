# This script leverages multiple models and features from Ultralytics, licensed under AGPL-3.0.
# See: https://github.com/ultralytics/ for more information.


import argparse
import cv2
from ultralytics import solutions


parser = argparse.ArgumentParser(description="Queue management")
parser.add_argument("-v", "--video_path", help="Video path")
arguments = parser.parse_args()
video_path = arguments.video_path


cap = cv2.VideoCapture(video_path)

assert cap.isOpened(), "Error reading video file"
w, h, fps = (
    int(cap.get(x))
    for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
)

video_writer = cv2.VideoWriter(
    "queue_management2.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
)

# Variable to store the queue region
queue_region = []


# Mouse callback function to draw the queue region
def draw_queue_region(event, x, y, flags, param):
    global queue_region
    if event == cv2.EVENT_LBUTTONDOWN:
        queue_region.append((x, y))


# Get the first frame from the video
success, first_frame = cap.read()
assert success, "Error reading the first frame of the video"

# Set up the window and bind the mouse callback
cv2.namedWindow("Draw Queue Region", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Draw Queue Region", draw_queue_region)

# Let the user draw the queue region
while True:
    temp_frame = first_frame.copy()
    if len(queue_region) > 1:
        for i in range(len(queue_region) - 1):
            cv2.line(temp_frame, queue_region[i], queue_region[i + 1], (0, 255, 0), 2)
        # Draw a line connecting the last point to the first point
        if len(queue_region) > 2:
            cv2.line(temp_frame, queue_region[-1], queue_region[0], (0, 255, 0), 2)

    cv2.namedWindow("Draw Queue Region", cv2.WINDOW_NORMAL)
    cv2.imshow("Draw Queue Region", temp_frame)
    key = cv2.waitKey(1) & 0xFF

    # Press 'q' to finish drawing
    if key == ord("q"):
        break

cv2.destroyWindow("Draw Queue Region")

# Initialize the QueueManager with the drawn region
queue = solutions.QueueManager(model="yolo11x.pt", classes=0, region=queue_region)

# Process the video frames
while cap.isOpened():
    success, im0 = cap.read()

    if success:
        out = queue.process_queue(im0)
        video_writer.write(im0)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    print("Video frame is empty or video processing has been successfully completed.")
    break

cap.release()
video_writer.release()
cv2.destroyAllWindows()
