from modules.mediapipe.mediapipe_FaceLandmarks import FaceDetector
from modules.mediapipe.ratio_calc import RatioCalculator
from modules.svm import SVM
from modules.fps import FPS
import cv2

labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

cap = cv2.VideoCapture(0)
fps = FPS()

face_model = FaceDetector(True, 10)
emotion_model = SVM('dataset/mediapipe_train_emotions.csv', labels)
emotion_model.train('emotions', scale=False, samples_limit=300, kernel='poly')

while True:
    _, frame = cap.read()
    h, w, _ = frame.shape

    FPS = fps.start()
    cv2.putText(frame, f'FPS:{int(FPS)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    faces = face_model.detect(frame)

    result_pred = []
    emotion_pred = []
    for i, face in enumerate(faces):
        ratio = RatioCalculator(face)
        result = emotion_model.predict(ratio.result())

        result = result.tolist()[0]
        result = 6 if result >= 6 else result

        result_pred.append(round(result, 5))
        emotion_pred.append(labels[round(result)])

        for j in range(0, 468):
            x = int(face[j].x * w)
            y = int(face[j].y * h)
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

        cv2.putText(frame, str(i), (int(face[10].x * w), int(face[10].y * h)), 0, 1, (0, 0, 255), 2)

    print(f'{result_pred} - {emotion_pred}')

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    cv2.imshow("main", frame)
