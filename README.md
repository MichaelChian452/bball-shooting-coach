# Basketball Shooting Coach

This project attempts to detect the user's basketball shooting form using [YOLO](https://github.com/ultralytics/ultralytics) object detection and [MediaPipe](https://github.com/google/mediapipe). SQLite is used as a backend database.



## Identifying a Basketball

A custom YOLOv8 object detection model was trained to identify a basketball within a given image or video.

<img src="https://github.com/MichaelChian452/bball-shooting-coach/assets/43621839/c0097319-9c26-419c-8976-19bd54db4f0f" alt="Indy and Basketball" width="250"/>

## Quantitative Data Collected

This program intends to help the user develop a more consistent shot. Data such as maximum knee bend, release angle, and release time are many of the helpful metrics that are detected from a live video feed.