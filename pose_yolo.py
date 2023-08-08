import os
from ultralytics import YOLO
import cv2
import numpy as np
import time

# -- Constants --

PRACC_DIR = os.path.join('.', 'pracc')

ball1_path = os.path.join(PRACC_DIR, 'cropped.jpg')

# -- Testing on image --

pose_model = YOLO('yolov8n-pose.pt')

# Open the webcam
cap = cv2.VideoCapture(0)

prev_frame_time = 0

new_frame_time = 0

while cap.isOpened():
    # Read frame from video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 pose estimation on frame
        results = pose_model(frame, save=True)

        # Get annotated frame
        annotated_frame = results[0].plot()

        new_frame_time = time.time()
        fps = 1/(new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)
        cv2.putText(annotated_frame, fps, (7, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('yolov8 estimate', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release the webcam and destroy the windows
cap.release()
cv2.destroyAllWindows()