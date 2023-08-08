import cv2
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
right_handed = 1 # 1 if right handed, 0 if left handed

ball_in_air = False
holding_ball = False
in_shooting_form = False

ball_pos = []
elbow = []
shoulder = []
knee = []

# Return true if person is holding basketball (not simply if their hand is touching but if they aren't in the process of dribbling)
def is_holding_ball(ball_center_pos, dom_wrist_pos) -> bool:
    ## For now, just pretend theyre holding ball
    # dist = np.hypot(ball_center_pos[0] - dom_wrist_pos[0], ball_center_pos[1] - dom_wrist_pos[1])
    # print('dist between ball and wrist: ' + str(dist))
    # holding_ball = dist < 300
    # return holding_ball
    return True

# Calculate angle between three points in the order a -- b -- c, for example, a = shoulder, b = elbow, c = wrist
def calculate_angle(a, b, c) -> float:
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle

def calculate_shooting_form(landmarks):
    dom_shoulder_pos = None
    dom_elbow_pos = None
    dom_wrist_pos = None
    dom_hip_pos = None
    dom_knee_pos = None
    dom_ankle_pos = None

    if right_handed:
        dom_shoulder_pos = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        dom_elbow_pos = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
        dom_wrist_pos = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
        dom_hip_pos = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        dom_knee_pos = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        dom_ankle_pos = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

    elbow_angle = calculate_angle(dom_shoulder_pos, dom_elbow_pos, dom_wrist_pos)
    shoulder_angle = calculate_angle(dom_elbow_pos, dom_shoulder_pos, dom_hip_pos)
    knee_angle = calculate_angle(dom_hip_pos, dom_knee_pos, dom_ankle_pos)

    if holding_ball or is_holding_ball(None, dom_wrist_pos):
        holding_ball = True
        


    return elbow_angle, shoulder_angle, knee_angle

# Open the webcam
cap = cv2.VideoCapture(0)

prev_frame_time = 0

new_frame_time = 0

elbow_angle = None

# Setup mediapipe
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        # Read frame from video
        success, frame = cap.read()

        if success:
            # Recolor frame from bgr channels to rbg for mediapipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Let mediapipe detect pose
            results = pose.process(image)

            # Recolor back to bgr for display by opencv
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract Landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                elbow_angle, shoulder_angle, knee_angle = calculate_shooting_form(landmarks)
            except:
                pass

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(image, 'elbow: ' + str(elbow_angle), (7, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow('mediapipe', image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

# Release the webcam and destroy the windows
cap.release()
cv2.destroyAllWindows()