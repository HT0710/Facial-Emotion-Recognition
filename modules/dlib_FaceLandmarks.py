import dlib
import os

path = os.path.dirname(__file__)
default = "shape_predictor_68_face_landmarks_GTX.dat"
print(f"Using GPU: {dlib.DLIB_USE_CUDA}\n")


class FaceDetector:
    def __init__(self, model=default, scale=1) -> None:
        self.__scale = scale
        self.__detector = dlib.get_frontal_face_detector()
        self.__predictor = dlib.shape_predictor(rf'{path}\{model}')
        self.__faceLandmarks = None

    def landmarks(self) -> (list or None):
        return self.__faceLandmarks

    def part(self, position: int) -> (tuple or None):
        return self.__faceLandmarks[position]

    def detect(self, image) -> list:
        self.__faceLandmarks = []
        faces = self.__detector(image, self.__scale)

        for face in faces:
            landmarks = self.__predictor(image, face)

            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y

                self.__faceLandmarks.append((x, y))

        return self.landmarks()
