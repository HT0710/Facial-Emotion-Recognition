from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.svm import SVR
import pandas as pd
import numpy as np


class SVM:
    def __init__(self, dataset: str, labels: list) -> None:
        """Support Vector Machines"""
        # Using SVR - Support Vector Prediction
        try:
            self.__dataset = pd.read_csv(dataset)
        except:
            raise print('Error: Reading dataset - Wrong file or Cannot find file path')
        self.__labels = labels
        self.__model = None

    @staticmethod
    def __features_scaling(data):
        """Returns the scaled features"""
        # Google for why used: Scaling and Normalization
        sc = StandardScaler()
        return sc.fit_transform(data)

    def __labels_transforming(self, data, labels, limit):
        """Return the transformed labels and drop data if limit"""
        # Change string labels to relative numbers and Limit the training input for data equality
        drop = []
        count = [0 for i in labels]

        limit = float('inf') if limit == 0 else limit
        for i, emotion in enumerate(labels):
            for j, label in enumerate(self.__labels):
                if emotion == label:
                    if count[j] >= limit:
                        drop.append(i)
                    else:
                        labels[i] = j
                        count[j] += 1

        X = np.delete(data, drop, 0)
        y = np.delete(labels, drop, 0)
        return X, y

    def train(self, predict_label: str, scale: bool = False, samples_limit: int = 0, kernel: str = 'rbf') -> None:
        """Train the SVM model

        Parameters
        ----------
        predict_label : str
            Label to predict
        scale : bool
            Scaling the data | default=None
        samples_limit : int
            Limit the input samples | default=None
        kernel : str
            {'poly', 'rbf'} | default='rbf'


        Returns
        -------
        -
            None
        """

        print(f"\nSetting -> Scale: {scale}, Limit: {samples_limit}, Kernel: {kernel}")

        dataset = self.__dataset

        X = dataset.loc[:, dataset.columns != predict_label].values
        y = dataset.loc[:, predict_label].values

        X = self.__features_scaling(X) if scale else X
        X, y = self.__labels_transforming(X, y, samples_limit)
        # print(Counter(y))

        print("Training data...")
        self.__model = SVR(kernel=kernel)
        self.__model.fit(X, y)
        print("Training - Done")

    def predict(self, data: list) -> float:
        return self.__model.predict([data])
