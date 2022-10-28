from mediapipe_FaceLandmarks import FaceDetector
from ratio_calc import RatioCalculator
import tensorflow as tf
import cv2
import os


datatype = 'train'
model = FaceDetector(True, 1)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
total = 7177 if datatype == 'test' else 28708
progbar = tf.keras.utils.Progbar(total)
loss = i = 0


with open(f'../../dataset/mediapipe_{datatype}_emotions.csv', 'a+') as f:
    for emotion in emotion_labels:
        current_path = f"../../dataset/{datatype}/{emotion}"
        for path, dirs, files in os.walk(current_path):
            for file in files:
                progbar.update(i)
                i += 1

                img = cv2.imread(f'{current_path}/{file}')
                imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, _ = img.shape

                faces = model.detect(imgRGB)
                if not faces:
                    loss += 1
                    continue

                for face in faces:
                    ratio = RatioCalculator(face)
                    dataset = ratio.result()
                    f.write(f"{''.join(f'{e},' for e in dataset)}{emotion}\n")


print(f"|- Detect: {total-loss}")
print(f"|- Loss: {loss}")
print(f'Efficiency: {round((total-loss)/total, 2)}%')
