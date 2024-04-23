import pandas as pd
import seaborn as sns
import numpy as np
from google.colab import drive

drive.mount('/content/drive')

import pandas as pd
# Heart disease Loading Data
df = "/content/drive/MyDrive/heart_data.csv"

df = pd.read_csv('/content/drive/MyDrive/heart_data.csv')
print(df.head())

# model.py

import pickle
from sklearn import linear_model

class HeartDiseaseModel:
    def __init__(self):
        self.model = None

    def train(self, X_train, y_train):
        self.model = linear_model.LinearRegression()
        self.model.fit(X_train, y_train)

    def save_model(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.model, file)

    def load_model(self, filename):
        with open(filename, 'rb') as file:
            self.model = pickle.load(file)

    def predict(self, X):
        return self.model.predict(X)
