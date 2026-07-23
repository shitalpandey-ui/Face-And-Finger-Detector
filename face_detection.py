"""
Run it with:
    python face_detection_simple.py
Press 'q' to quit.
"""

import os
import urllib.request
import cv2


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
    print("Done!")
    return CASCADE_FILE

face_recipe_path = get_face_recipe()
face_detector = cv2.CascadeClassifier(face_recipe_path)

webcam = cv2.VideoCapture(0)

print("Webcam is on. Press 'q' at any time to quit.")


while True:

    success, frame = webcam.read()

    if not success:
        print("Couldn't read from webcam. Stopping.")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces_found = face_detector.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,   # how carefully it checks different face sizes
        minNeighbors=5,    # how sure it needs to be before saying "yes, a face"
        minSize=(40, 40),  # ignore anything smaller than 40x40 pixels
    )

    for (x, y, width, height) in faces_found:
        top_left = (x, y)
        bottom_right = (x + width, y + height)
        green = (0, 255, 0)
        thickness = 2
        cv2.rectangle(frame, top_left, bottom_right, green, thickness)

    text = f"Faces detected: {len(faces_found)}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Face Detection - press q to quit", frame)

    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord("q"):
        break

webcam.release()        
cv2.destroyAllWindows()   
print("Stopped.")