import cv2
import time

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

# press q to exit
while True:
    # time.sleep(0.1)

    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.5, minNeighbors=5)

    for (x, y, w, h) in faces:
        width = x + w
        height = y + h

        roi_frame = frame[x:x + h, y:y + h]
        cv2.rectangle(frame, (x, y), (width, height), (0, 0, 255), thickness=3)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    cv2.imshow("main", frame)

cap.release()
cv2.destroyAllWindows()

#%%
