# Diabetes Prediction Model

#importing libraries

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

# loading the dataset
df = pd.read_csv('diabetes.csv')

print(df.head())

print(df.shape)

print(df.describe())

cat= df['Outcome'].value_counts()
print(cat)

print(df.groupby('Outcome').mean())

# separating data and labels
X = df.drop(columns='Outcome', axis=1)
Y = df['Outcome']   
print(X)
print(Y)


