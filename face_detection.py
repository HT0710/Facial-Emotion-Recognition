import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

face_cascade1 = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade2 = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
face_cascade3 = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
face_model = [face_cascade1, face_cascade2, face_cascade3]

cap = cv2.VideoCapture(0)

def check_face(frame, i = 0):
    if (i == len(face_model)):
        return (), ""
    
    faces = face_model[i].detectMultiScale(gray_frame, scaleFactor=1.5, minNeighbors=5)
    if (type(faces) != tuple):
        return faces, i
    else:
        i += 1
        return check_face(frame, i)
    

while True:
    time.sleep(0.1)

    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces, ftype = check_face(gray_frame)
    print(ftype)
    
    for (x, y, w, h) in faces:
        width = x + w
        height = y + h

        roi_frame = frame[x:x+h, y:y+h]
        cv2.rectangle(frame, (x, y), (width, height), (0, 0, 255), thickness=3)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

    cv2.imshow("main", frame)

cap.release()
cv2.destroyAllWindows()
