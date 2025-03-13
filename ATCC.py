import sys
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Get video paths from command-line arguments
video_paths = sys.argv[1:]  # Expecting 4 video paths

if len(video_paths) != 4:
    print("Error: Please provide exactly 4 video files as input.")
    sys.exit(1)

# Open video captures
caps = [cv2.VideoCapture(path) if path else None for path in video_paths]

# Frame size
FRAME_WIDTH, FRAME_HEIGHT = 640, 360

# Vehicle count tracking
vehicle_counts = [0] * len(caps)
vehicle_classifications = [{} for _ in range(len(caps))]

# COCO class names (Only relevant vehicle classes)
coco_classes = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# Create full-screen window
cv2.namedWindow("Traffic Management System", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Traffic Management System", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while any(cap.isOpened() for cap in caps):
    frames = []
    total_classifications = {c: 0 for c in coco_classes.values()}
    total_vehicle_count = 0

    for i, cap in enumerate(caps):
        if cap is None:
            frames.append(np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8))
            continue

        ret, frame = cap.read()
        if not ret:
            frames.append(np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8))
            continue

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        results = model(frame)

        vehicle_count = 0
        classifications = {}

        for obj in results[0].boxes.data:
            x1, y1, x2, y2, confidence, class_id = obj[:6]
            x1, y1, x2, y2, class_id = map(int, [x1, y1, x2, y2, class_id])
            if class_id in coco_classes and confidence > 0.5:
                vehicle_count += 1
                vehicle_type = coco_classes[class_id]
                classifications[vehicle_type] = classifications.get(vehicle_type, 0) + 1
                total_classifications[vehicle_type] += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{vehicle_type} {confidence:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        vehicle_counts[i] = vehicle_count
        vehicle_classifications[i] = classifications
        total_vehicle_count += vehicle_count

        cv2.putText(frame, f"Count: {vehicle_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        frames.append(frame)

    max_index = np.argmax(vehicle_counts)  # Find the lane with the highest traffic

    # Draw traffic signals
    for i, frame in enumerate(frames):
        red_color = (0, 0, 255) if i != max_index else (255, 255, 255)
        green_color = (0, 255, 0) if i == max_index else (255, 255, 255)

        cv2.circle(frame, (50, 80), 15, red_color, -1)  # Red light
        cv2.circle(frame, (50, 110), 15, green_color, -1)  # Green light

    # Summary panel
    summary_frame = np.zeros((70, FRAME_WIDTH * len(frames), 3), dtype=np.uint8)
    total_count_text = f"Total Vehicles: {total_vehicle_count}"
    class_text = " | ".join([f"{v}: {total_classifications.get(v, 0)}" for v in coco_classes.values()])
    cv2.putText(summary_frame, total_count_text, (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(summary_frame, class_text, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Arrange frames
    if len(frames) == 1:
        final_display = frames[0]
    elif len(frames) == 2:
        final_display = np.hstack(frames)
    else:
        mid = len(frames) // 2
        top_row = np.hstack(frames[:mid])
        bottom_row = np.hstack(frames[mid:]) if len(frames) > mid else np.zeros_like(top_row)
        final_display = np.vstack([top_row, bottom_row])

    summary_frame = cv2.resize(summary_frame, (final_display.shape[1], 70))
    final_display = np.vstack([summary_frame, final_display])

    cv2.imshow("Traffic Management System", final_display)

    # Press 'Q' or 'ESC' to exit
    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:
        break

# Release video captures
for cap in caps:
    if cap:
        cap.release()
cv2.destroyAllWindows()

