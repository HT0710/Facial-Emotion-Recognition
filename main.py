import cv2
import numpy as np
import dlib
import time

cap = cv2.VideoCapture("videos/1.mp4")
pTime = 0

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    r'C:\Users\HT0710\Documents\GitHub\Facial-Emotion-Recognition\shape_predictor_68_face_landmarks.dat')

while True:
    _, frame = cap.read()
    frame = cv2.resize(frame, (0, 0))

    cTime = time.time()
    try:
        fps = 1 / (cTime - pTime)
    except:
        pass
    pTime = cTime
    cv2.putText(frame, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    print(int(fps))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        landmarks = predictor(gray, face)

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.putText(frame, str(n), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    cv2.imshow("main", frame)
