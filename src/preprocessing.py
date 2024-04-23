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

# Data Preprocessing
x_df = df.drop('heart.disease', axis=1)
y_df = df['heart.disease']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.3, random_state=42)

# Model Training
from sklearn import linear_model

# Create Linear Regression object
model = linear_model.LinearRegression()

# Train the model using independent variables
model.fit(X_train, y_train)

# Print the R^2 value
print(model.score(X_train, y_train))
