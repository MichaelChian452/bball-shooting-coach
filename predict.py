import os
from ultralytics import YOLO
import cv2

# -- Constants --

PRACC_DIR = os.path.join('.', 'pracc')

ball1_path = os.path.join(PRACC_DIR, 'ball2_40.jpg')

frame = cv2.imread(ball1_path)

# Model with 25 epochs
# model_path = os.path.join('.', 'runs', 'detect', 'train9', 'weights', 'best.pt')

# Model with 150 epochs
model_path = os.path.join('.', 'best.pt')

# -- Testing on image --

# model = YOLO(model_path)
# results = model(frame)[0]
# threshold = 0.5

# for result in results.boxes.data.tolist():
#     x1, y1, x2, y2, score, class_id = result

#     if score > threshold:
#         cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
#         cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

# cv2.imshow('cat', frame)
# cv2.waitKey(0)

# -- Testing on video --

# video_path = os.path.join(PRACC_DIR, 'IMG_4354.mov')
# video_path_out = '{}_out.mov'.format(video_path)

# cap = cv2.VideoCapture(video_path)
# ret, frame = cap.read()
# H, W, _ = frame.shape
# out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

# model = YOLO(model_path)

# threshold = 0.5

# while ret:

#     results = model(frame)[0]

#     for result in results.boxes.data.tolist():
#         x1, y1, x2, y2, score, class_id = result

#         if score > threshold:
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
#             cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

#     out.write(frame)
#     ret, frame = cap.read()

# cap.release()
# out.release()
# cv2.destroyAllWindows()