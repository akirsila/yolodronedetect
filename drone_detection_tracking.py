import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv8 model
model = YOLO("yolov8n-drone.pt")  # Ensure this file exists in your directory

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

# Open video stream (use 0 for webcam)
cap = cv2.VideoCapture("test.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame.")
        break

    # Run YOLO detection
    results = model(frame)

    detections = []
    confidences = []
    
    print(f"Detected {len(results[0].boxes)} objects")  # Debugging

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Convert bounding box to integers
            score = box.conf[0].item()  # Get confidence score
            cls = int(box.cls[0].item())  # Convert class ID to integer

            print(f"Detection: x1={x1}, y1={y1}, x2={x2}, y2={y2}, Score={score}, Class={cls}")  # Debugging

            # Adjust class filtering based on your dataset (remove class filtering for now)
            if score > 0.3:  # Lower confidence threshold for testing
                detections.append(([x1, y1, x2 - x1, y2 - y1], score))
                confidences.append(score)

    if len(detections) == 0:
        print("No valid detections found.")  # Debugging

    # Update DeepSORT tracker
    tracked_objects = tracker.update_tracks(detections, frame=frame)

    for track in tracked_objects:
        if not track.is_confirmed():
            continue

        x1, y1, x2, y2 = track.to_ltrb()
        obj_id = track.track_id
        center_x, center_y = int((x1 + x2) // 2), int((y1 + y2) // 2)

        print(f"Tracking ID {obj_id}: x1={x1}, y1={y1}, x2={x2}, y2={y2}")  # Debugging

        # Draw bounding box and tracking ID
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {obj_id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw crosshair for aiming
        cv2.drawMarker(frame, (center_x, center_y), (0, 0, 255), markerType=cv2.MARKER_CROSS, thickness=2)

    cv2.imshow("Drone Detection & Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
