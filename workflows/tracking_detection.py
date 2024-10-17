# This script leverages multiple models and features from Ultralytics, licensed under AGPL-3.0.
# See: https://github.com/ultralytics/ for more information.

"""
Available Trackers

Ultralytics YOLO supports the following tracking algorithms. They can be enabled by passing the relevant YAML configuration file such as tracker=tracker_type.yaml:

    BoT-SORT - Use botsort.yaml to enable this tracker.
    ByteTrack - Use bytetrack.yaml to enable this tracker.

"""

import argparse
import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

parser = argparse.ArgumentParser(description="Heatmap line counting")
parser.add_argument("-v", "--video_path", help="Video path")
arguments = parser.parse_args()
video_path = arguments.video_path


# Load the YOLOv8 model
# model = YOLO('yolov8n.pt')  # Load an official Detect model
model = YOLO("yolov8x.pt")
# model = YOLO('yolov8n-seg.pt')  # Load an official Segment model
# model = YOLO('yolov8n-pose.pt')  # Load an official Pose model
# model = YOLO('path/to/best.pt')  # Load a custom trained model

cap = cv2.VideoCapture(video_path)

# Video writer
video_writer = cv2.VideoWriter(
    "tracking_output.avi",
    cv2.VideoWriter_fourcc(*"mp4v"),
    int(cap.get(5)),
    (int(cap.get(3)), int(cap.get(4))),
)


# Store the track history
track_history = defaultdict(lambda: [])

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(
            frame, persist=True, conf=0.3, iou=0.5, show=False, tracker="botsort.yaml"
        )  # bytetrack.yaml or botsort.yaml

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 120:  # retain 90 tracks for 90 frames #30
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                annotated_frame,
                [points],
                isClosed=False,
                color=((120, 214, 12)),
                thickness=4,
            )

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)
        video_writer.write(annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
video_writer.release()
cv2.destroyAllWindows()
