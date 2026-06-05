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
X = df.drop(columns='Outcome')
Y = df['Outcome']   
print(X)
print(Y)

# data standardization

scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)

print(standardized_data)

X = standardized_data
Y = df['Outcome']

print(X)
print(Y)


# train test split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)


# training the model

classifier = svm.SVC(kernel='linear')
#training svm classifier
classifier.fit(X_train, Y_train)


# model evaluation

#accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data is ', training_data_accuracy)

#accuracy score on the testing data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data is ', test_data_accuracy)


# making a predictive system

input_data = (0,85,127,22,82,22.4,0.08,19)


# changing the input data to a numpy array

input_data_reshaped = np.asarray(input_data).reshape(1, -1)
input_data_scaled = scaler.transform(input_data_reshaped)

# standardize the input data

print(input_data_scaled)

prediction = classifier.predict(input_data_scaled)
print(prediction)

if (prediction[0] == 0):
    print('The person is not diabetic')
else:
    print('The person is diabetic') 

