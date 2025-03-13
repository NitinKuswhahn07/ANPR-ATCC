import sys
import cv2
import os
import numpy as np
from ultralytics import YOLO

# Suppress terminal on Windows (for pythonw.exe)
if os.name == "nt":
    import ctypes
    ctypes.windll.kernel32.FreeConsole()

# Check if a video file path is provided as an argument; otherwise, use default.
video_path = sys.argv[1] if len(sys.argv) > 1 else "emergency_vehicle_detection.mp4"

# Open the video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Unable to open video file {video_path}")
    sys.exit(1)  # Exit with error code

# Load YOLO model
model = YOLO('yolov8n.pt')

# Function to check if the detected vehicle is mostly black or blue
def is_black_or_blue_vehicle(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Define black color range in HSV
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])

    # Define blue color range in HSV
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([140, 255, 255])

    # Create masks for black and blue
    black_mask = cv2.inRange(hsv, lower_black, upper_black)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Calculate the proportion of black and blue pixels
    black_ratio = cv2.countNonZero(black_mask) / (roi.shape[0] * roi.shape[1])
    blue_ratio = cv2.countNonZero(blue_mask) / (roi.shape[0] * roi.shape[1])

    # If more than 50% of the vehicle is black or blue, it's considered a black/blue vehicle
    return black_ratio > 0.6 or blue_ratio > 0.6

# Create a resizable OpenCV window
cv2.namedWindow("Emergency Vehicle Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Emergency Vehicle Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection on the frame
    results = model(frame)

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            class_name = model.names[cls]
            
            if class_name in ['truck']:  # Detect emergency vehicles (fire truck, ambulance)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Extract the detected vehicle region
                roi = frame[y1:y2, x1:x2]

                # Skip the vehicle if it's black or blue
                if is_black_or_blue_vehicle(roi):
                    continue

                # Draw red bounding box for emergency vehicles
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "Emergency Vehicle", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Emergency Vehicle Detection", frame)

    # Press 'q' or 'ESC' to exit
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:  # 27 is ESC key
        break

cap.release()
cv2.destroyAllWindows()
