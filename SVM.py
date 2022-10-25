import numpy as np
import pandas as pd
from collections import Counter
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data_path = 'dataset/train_emotions.csv'
labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

dataset = pd.read_csv(data_path)

X = dataset.loc[:, dataset.columns != 'emotions'].values
y = dataset.loc[:, 'emotions'].values

for i, emotion in enumerate(y):
    for j, label in enumerate(labels):
        if emotion == label:
            y[i] = j

sc = StandardScaler()
X = sc.fit_transform(X)

model = SVR()
model.fit(X, y)

pred = model.predict([[1.161,1.5,1.0952,1.3809,1.0332,1.821,1.962,1.5205]])

print(pred)