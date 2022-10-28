from dlib_FaceLandmarks import FaceDetector
from ratio_calc import RatioCalculator
import tensorflow as tf
import numpy as np
import time
import cv2
import os

datatype = 'test'
model = FaceDetector(scale=1)
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
total = 7177 if datatype == 'test' else 28708
progbar = tf.keras.utils.Progbar(total)
start_time = time.time()
loss = i = 0

with open(f'../../dataset/dlib_{datatype}_emotions.csv', 'a+') as f:
    for emotion in emotion_labels:
        current_path = f"../../dataset/{datatype}/{emotion}"
        for path, dirs, files in os.walk(current_path):
            for file in files:
                progbar.update(i)
                i += 1

                img = cv2.imread(f'{current_path}/{file}')

                face = model.detect(img)
                if not face:
                    loss += 1
                    continue

                ratio = RatioCalculator(face)
                dataset = ratio.result()
                f.write(f"{''.join(f'{e},' for e in dataset)}{emotion}\n")


print(f"\nExecution time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
print(f"|- Detect: {total-loss}")
print(f"|- Loss: {loss}")
print(f'Efficiency: {round((total-loss)/total, 2)}%')

