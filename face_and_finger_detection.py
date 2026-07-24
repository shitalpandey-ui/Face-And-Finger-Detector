

import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


CASCADE_FILE = "haarcascade_frontalface_default.xml"

def get_face_recipe():
    if os.path.exists(CASCADE_FILE):
        return CASCADE_FILE
    print("Downloading the face detection file (only happens once)...")
    url = (
        "https://raw.githubusercontent.com/opencv/opencv/4.x/data/"
        "haarcascades/haarcascade_frontalface_default.xml"
    )
    urllib.request.urlretrieve(url, CASCADE_FILE)
    return CASCADE_FILE


def detect_faces(gray_frame, face_detector):
    return face_detector.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )


def draw_faces(frame, faces_found):
    for (x, y, width, height) in faces_found:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
    return len(faces_found)


MODEL_FILE = "hand_landmarker.task"


def get_hand_model():
    if os.path.exists(MODEL_FILE):
        return MODEL_FILE
    return MODEL_FILE


FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_JOINTS_BELOW_TIP = [3, 6, 10, 14, 18]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          
    (0, 5), (5, 6), (6, 7), (7, 8),         
    (5, 9), (9, 10), (10, 11), (11, 12),     
    (9, 13), (13, 14), (14, 15), (15, 16),   
    (13, 17), (17, 18), (18, 19), (19, 20), 
    (0, 17),                                 
]


def count_raised_fingers(hand_points, handedness_label):
    fingers_up = []

   
    thumb_tip_x = hand_points[FINGER_TIPS[0]].x
    thumb_joint_x = hand_points[FINGER_JOINTS_BELOW_TIP[0]].x
    if handedness_label == "Right":
        thumb_is_up = thumb_tip_x < thumb_joint_x
    else:
        thumb_is_up = thumb_tip_x > thumb_joint_x
    fingers_up.append(1 if thumb_is_up else 0)

    for tip_id, joint_id in zip(FINGER_TIPS[1:], FINGER_JOINTS_BELOW_TIP[1:]):
        finger_is_up = hand_points[tip_id].y < hand_points[joint_id].y
        fingers_up.append(1 if finger_is_up else 0)

    return fingers_up


def draw_hand_skeleton(frame, hand_points, frame_width, frame_height):
    pixel_points = [
        (int(p.x * frame_width), int(p.y * frame_height)) for p in hand_points
    ]
    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(frame, pixel_points[start_idx], pixel_points[end_idx], (255, 255, 255), 2)
    for point in pixel_points:
        cv2.circle(frame, point, 5, (0, 165, 255), -1)


face_detector = cv2.CascadeClassifier(get_face_recipe())

hand_landmarker = mp_vision.HandLandmarker.create_from_options(
    mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=get_hand_model()),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
)

webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_BUFFERSIZE, 1)   
print("Webcam is on. Show your face and hands! Press 'q' to quit.")

start_time = time.time()

while True:
    success, frame = webcam.read()
    if not success:
        break

    success, frame = webcam.read()
    frame = cv2.resize(frame, (640, 480))   
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_found = detect_faces(gray_frame, face_detector)
    face_count = draw_faces(frame, faces_found)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int((time.time() - start_time) * 1000)
    hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

    total_fingers_up = 0
    if hand_result.hand_landmarks:
        for hand_points, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
            hand_label = handedness[0].category_name
            draw_hand_skeleton(frame, hand_points, frame_width, frame_height)

            fingers_up_this_hand = sum(count_raised_fingers(hand_points, hand_label))
            total_fingers_up += fingers_up_this_hand

            wrist = hand_points[0]
            text_position = (
                int(wrist.x * frame_width) - 20,
                int(wrist.y * frame_height) + 40,
            )
            cv2.putText(
                frame, f"{hand_label}: {fingers_up_this_hand}", text_position,
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2,
            )

    cv2.putText(frame, f"Faces: {face_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Fingers up: {total_fingers_up}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

    cv2.imshow("Face + Finger Detection - press q to quit", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

webcam.release()
cv2.destroyAllWindows()
hand_landmarker.close()
print("Stopped.")