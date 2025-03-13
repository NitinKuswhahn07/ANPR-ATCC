import sys
import cv2
import numpy as np
import time
import datetime
import os
import torch
import json
from collections import defaultdict
from ultralytics import YOLO
from pathlib import Path

# Suppress terminal on Windows (for pythonw.exe)
if os.name == "nt":
    import ctypes
    ctypes.windll.kernel32.FreeConsole()

# Get video file path from arguments or use default
video_path = sys.argv[1] if len(sys.argv) > 1 else "speed_test.mp4"

# Open the video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Unable to open video file {video_path}")
    sys.exit(1)

# Load YOLO model
model = YOLO("yolov8n.pt")

# Speed limit in km/h
SPEED_LIMIT = 20
DISTANCE = 5  # Distance covered between detection lines (in meters)
CONFIDENCE_THRESHOLD = 0.3
FRAME_SKIP = 1

# Detection line positions (percentage of frame height)
DETECTION_LINES = [0.3, 0.7]
VEHICLE_CLASSES = [2, 3, 5, 7]  # COCO: Car, Motorcycle, Bus, Truck

# Dictionary to track vehicle speed calculations
vehicle_tracks = defaultdict(dict)

# OpenCV window setup
cv2.namedWindow("Overspeeding Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Overspeeding Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def calculate_speed(time1, time2, y1, y2):
    try:
        time_diff = abs(time2 - time1)
        if 0 < time_diff < 1.0:
            distance = abs(y2 - y1) * DISTANCE / 100
            speed = (distance / time_diff) * 3.6  # Convert m/s to km/h
            return speed if 10 <= speed <= 150 else 0
    except:
        pass
    return 0

while True:
    success, frame = cap.read()
    if not success:
        break

    height, width = frame.shape[:2]
    frame = cv2.resize(frame, (640, 480))
    height, width = frame.shape[:2]

    # Draw detection lines
    lines = [int(height * pos) for pos in DETECTION_LINES]
    for line in lines:
        cv2.line(frame, (0, line), (width, line), (255, 0, 0), 2)

    # Run YOLO model on frame
    results = model.track(
        frame, persist=True, conf=CONFIDENCE_THRESHOLD, classes=VEHICLE_CLASSES, verbose=False
    )

    if not results or not results[0].boxes:
        cv2.imshow("Overspeeding Detection", frame)
        if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:  # Quit on 'q' or ESC
            break
        continue

    current_time = time.time()

    for box in results[0].boxes:
        if not box.id:
            continue

        vehicle_id = int(box.id)
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        center_y = (y1 + y2) / 2

        # First detection of the vehicle
        if vehicle_id not in vehicle_tracks:
            vehicle_tracks[vehicle_id] = {'time': current_time, 'position': center_y}
        else:
            first_detection = vehicle_tracks[vehicle_id]
            speed = calculate_speed(
                first_detection['time'], current_time, first_detection['position'], center_y
            )

            if speed > SPEED_LIMIT:
                color = (0, 0, 255)  # Red for overspeeding
            else:
                color = (0, 255, 0)  # Green for normal speed

            # Draw bounding box and speed text
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{speed:.1f} km/h", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Update the vehicle's tracking data
            vehicle_tracks[vehicle_id] = {'time': current_time, 'position': center_y}

    # Show output frame
    cv2.imshow("Overspeeding Detection", frame)

    # Quit on 'q' or ESC
    if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
