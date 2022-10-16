import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

face_cascade1 = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade2 = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
face_cascade3 = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture(0)
ftype = None

while True:
	ret, frame = cap.read()
	time.sleep(0.1)
	print(ftype)
	gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	ftype = 0
	faces = face_cascade1.detectMultiScale(gray_frame, scaleFactor=1.5, minNeighbors=5)
	if (type(faces) == tuple):
		ftype = 1
		faces = face_cascade2.detectMultiScale(gray_frame, scaleFactor=1.5, minNeighbors=5)
		if (type(faces) == tuple):
				ftype = 2
				faces = face_cascade3.detectMultiScale(gray_frame, scaleFactor=1.5, minNeighbors=5)
        
	for (x, y, w, h) in faces:
		width = x + w
		height = y + h
		
		roi_frame = frame[x:x+h, y:y+h]
		cv2.rectangle(frame, (x, y), (width, height), (0, 0, 255), thickness=3)
		
	if cv2.waitKey(10) & 0xFF == ord(''):
		break
	
	cv2.imshow("main frame", frame)

cap.release()
cv2.destroyAllWindows()
