import cv2
import mediapipe as mp


class FaceDetector:
    def __init__(self, static=False, max_face=1):
        self.__mpFaceMesh = mp.solutions.face_mesh
        self.__faceMesh = self.__mpFaceMesh.FaceMesh(static_image_mode=static, max_num_faces=max_face)
        self.__faceLandmarks = None

    def landmarks(self) -> (list or None):
        return self.__faceLandmarks

    def detect(self, image) -> (list or None):
        self.__faceLandmarks = []

        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        facial = self.__faceMesh.process(imgRGB)
        landmarks = facial.multi_face_landmarks

        if landmarks is None:
            return self.landmarks()

        for i, facial_landmarks in enumerate(landmarks):
            self.__faceLandmarks.append([])
            for j in range(0, 468):
                self.__faceLandmarks[i].append(facial_landmarks.landmark[j])

        return self.landmarks()
