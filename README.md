# Basketball Shooting Coach

Trying to improve your basketball shot is a daunting task. Coaches and videos give you the basic idea about shooting form but they cannot always be there to help you. This program will help a player quantify their shot to improve accuracy and achieve consistent body mechanics through collecting data such as maximum knee bend, release angle, and release time from a live video feed.

This project attempts to detect the user's basketball shooting form using [YOLO](https://github.com/ultralytics/ultralytics) object detection and [MediaPipe](https://github.com/google/mediapipe) pose estimation. SQLite is used as a backend database to store data about each shot attempt.

## Identifying a Basketball

A custom YOLOv8 object detection model was trained to identify a basketball within a given image or video. The model was trained with an image dataset [provided on roboflow here](https://universe.roboflow.com/eagle-eye/basketball-1zhpe) by EagleEye.

<img src="https://github.com/MichaelChian452/bball-shooting-coach/assets/43621839/c0097319-9c26-419c-8976-19bd54db4f0f" alt="Indy and Basketball" width="250"/>


## Identifying the Shooter

MediaPipe Pose's pose estimation ML model is used in conjuction with the basketball model to detect and quantify the player's shooting form.

<img src="https://camo.githubusercontent.com/e1ac1f6f369d41ea9de9d21bc74f9de89c2ed892d9bcc2554acac8041fa571d6/68747470733a2f2f6d65646961706970652e6465762f696d616765732f6d6f62696c652f706f73655f747261636b696e675f6578616d706c652e676966" alt="Person stretching" width="250"/>

MediaPipe example.