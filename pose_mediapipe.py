import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np
import time

class ShootingCoach:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.bball_model = YOLO('best.pt')
        self.height = -1
        self.width = -1

        self.right_handed = 1 # 1 if right handed, 0 if left handed

        self.prev_frame_time = 0

        self.new_frame_time = 0

        self.ball_in_air = False
        self.holding_ball = False
        self.prev_in_shooting_form = False
        self.in_shooting_form = False

        self.angles = None
        self.landmarks = None

        self.dom_shoulder_pos = None
        self.dom_elbow_pos = None
        self.dom_wrist_pos = None
        self.dom_hip_pos = None
        self.dom_knee_pos = None
        self.dom_ankle_pos = None
        self.avg_dom_hand_pos = None # average position of wrist, index, pinky, and thumb

        self.prev_ball_pos = []
        self.ball_pos = []

        self.ball_positions = []
        self.elbow = []
        self.shoulder = []
        self.knee = []

    # In order for the person to be shooting, the ball should be going upwards (-y direction)
    def is_ball_going_up(self, prev_ball_center_pos, ball_center_pos):
        return True if prev_ball_center_pos[1] > ball_center_pos[1] else False

    # Return true if person is holding basketball (not simply if their hand is touching but if they aren't in the process of dribbling)
    def is_holding_ball(self, ball_center_pos, ball_width, avg_hand_pos) -> bool:
        if len(ball_center_pos) == 0:
            return False
        dist = np.hypot(ball_center_pos[0] - avg_hand_pos[0], ball_center_pos[1] - avg_hand_pos[1])
        holding_ball = dist < 4 * ball_width / 5 # should be half ball width technically but giving some leeway
        return holding_ball

    # Calculate angle between three points in the order a -- b -- c, for example, a = shoulder, b = elbow, c = wrist
    def calculate_angle(self, a, b, c) -> float:
        if a is None or b is None or c is None:
            return None
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle
        return angle

    def calculate_shooting_form(self, bball_results, landmarks):
        # find position of ball and convert to relative coords
        self.prev_ball_pos = self.ball_pos
        self.ball_pos = []
        ball_width = 0
        for ball in bball_results:
            for box in ball.boxes.xyxy:
                x1, y1, x2, y2 = box[:4]
                ball_x_center = (x1 + x2) / 2 / self.width
                ball_y_center = (y1 + y2) / 2 / self.height
                self.ball_pos = [ball_x_center, ball_y_center]
                ball_width = np.abs((x2 - x1) / self.width)

        # find pose landmarks
        if self.right_handed:
            self.dom_shoulder_pos = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            self.dom_elbow_pos = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            self.dom_wrist_pos = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            self.dom_hip_pos = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            self.dom_knee_pos = [landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            self.dom_ankle_pos = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            self.avg_dom_hand_pos = [((landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x + landmarks[self.mp_pose.PoseLandmark.RIGHT_THUMB.value].x + 
                                       landmarks[self.mp_pose.PoseLandmark.RIGHT_PINKY.value].x + landmarks[self.mp_pose.PoseLandmark.RIGHT_INDEX.value].x) / 4), 
                                       ((landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y + landmarks[self.mp_pose.PoseLandmark.RIGHT_THUMB.value].y + 
                                       landmarks[self.mp_pose.PoseLandmark.RIGHT_PINKY.value].y + landmarks[self.mp_pose.PoseLandmark.RIGHT_INDEX.value].y) / 4)]

        elbow_angle = self.calculate_angle(self.dom_shoulder_pos, self.dom_elbow_pos, self.dom_wrist_pos)
        shoulder_angle = self.calculate_angle(self.dom_elbow_pos, self.dom_shoulder_pos, self.dom_hip_pos)
        knee_angle = self.calculate_angle(self.dom_hip_pos, self.dom_knee_pos, self.dom_ankle_pos)

        # check if person is holding the ball, updates boolean value accordingly
        if self.is_holding_ball(self.ball_pos, ball_width, self.avg_dom_hand_pos):
            self.holding_ball = True
            print('holding ball')
        else:
            self.holding_ball = False
            print('not holding')

        self.prev_in_shooting_form = self.in_shooting_form

        # if you are holding ball but the ball isnt giong up, it isnt a shot, so reset variables
        if self.holding_ball and not self.is_ball_going_up(self.prev_ball_pos, self.ball_pos):
            self.in_shooting_form = False
            self.shoulder = []
            self.elbow = []
            self.knee = []
        elif self.holding_ball: # but if the ball is going up, then it might be a shot, so keep track of variables
            self.in_shooting_form = True
            self.elbow.append(elbow_angle)
            self.shoulder.append(shoulder_angle)
            self.knee.append(knee_angle)
        elif self.prev_in_shooting_form and self.is_ball_going_up(self.prev_ball_pos, self.ball_pos): # if used to be shooting and ball is going up, then the shot is in the air
            self.ball_in_air = True
            

        return elbow_angle, shoulder_angle, knee_angle

    def shot_detection(self):
        # Open the webcam
        cap = cv2.VideoCapture(0)

        prev_frame_time = 0

        new_frame_time = 0

        self.elbow_angle = None
        self.shoulder_angle = None
        self.knee_angle = None


        # Setup mediapipe
        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                self.width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                self.height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                # Read frame from video
                success, frame = cap.read()

                if success:
                    # Recolor frame from bgr channels to rbg for mediapipe
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False

                    # Let mediapipe and yolo detect pose
                    pose_results = pose.process(image)
                    bball_results = self.bball_model(frame, save=True)

                    # Recolor back to bgr for display by opencv
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    # Get annotated frame with bball
                    image = bball_results[0].plot()
                    # Get annotated frame with pose
                    self.mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)


                    # Extract Landmarks
                    try:
                        self.landmarks = pose_results.pose_landmarks.landmark
                    except:
                        pass
                    self.angles = self.calculate_shooting_form(bball_results, self.landmarks)
                    elbow_angle = self.angles[0]
                    shoulder_angle = self.angles[1]
                    knee_angle = self.angles[2]
                    cv2.putText(image, 'elbow: ' + str(elbow_angle), (7, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, 'holding: ' + str(self.holding_ball), (7, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, 'in shoot: ' + str(self.in_shooting_form), (7, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, 'prev in shoot: ' + str(self.prev_in_shooting_form), (7, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 0), 1, cv2.LINE_AA)
                
                    # Add elbow angle text and fps
                    new_frame_time = time.time()
                    fps = 1/(new_frame_time - prev_frame_time)
                    prev_frame_time = new_frame_time
                    fps = int(fps)
                    fps = str(fps)
                    cv2.putText(image, fps, (7, 70), cv2.FONT_HERSHEY_COMPLEX, 0.5, (100, 255, 0), 1, cv2.LINE_AA)
                    
                    # Show image
                    cv2.imshow('mediapipe', image)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                else:
                    break

        # Release the webcam and destroy the windows
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    detector = ShootingCoach()
    detector.shot_detection()