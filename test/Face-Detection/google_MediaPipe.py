import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(static_image_mode=True, max_num_faces=50)

cap = cv2.VideoCapture(1)
ptime = 0

while True:
    _, frame = cap.read()
    ctime = time.time()
    try:
        fps = 1 / (ctime - ptime)
    except:
        pass
    ptime = ctime
    cv2.putText(frame, f"{int(fps)}", (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(frame, faceLms, mpFaceMesh.FACEMESH_CONTOURS)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    cv2.imshow("main", frame)
