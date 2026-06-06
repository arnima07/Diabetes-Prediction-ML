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

def line():
    print("="*60)
line()
print("DIABETES PREDICTION USING MACHINE LEARNING")
line()


#print(df.head())

#print(df.shape)

#print(df.describe())

cat= df['Outcome'].value_counts()
#print(cat)

#print(df.groupby('Outcome').mean())

# separating data and labels
X = df.drop(columns='Outcome')
Y = df['Outcome']   
#print(X)
#print(Y)

# data standardization

scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)

#print(standardized_data)

X = standardized_data
Y = df['Outcome']

#print(X)
#print(Y)


# train test split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
#print(X.shape, X_train.shape, X_test.shape)


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

input_data = (0,79.5,96,19,82,20.2,1.23,21)

print("\nPATIENT DETAILS")
print("-"*35)
print(f"Pregnancies = 0")
print(f"Glucose = 79.5")
print(f"BloodPressure = 96")
print(f"SkinThickness = 19")
print(f"Insulin = 82")
print(f"BMI = 20.2")
print(f"DiabetesPedigreeFunction = 1.23")
print(f"Age = 21")


print("\nDIABETES PEDIGREE FUNCTION SCALE")
print("-"*45)
print("0.078  -> Very Low Risk")
print("0.351  -> Low to Moderate Risk")
print("0.627  -> Moderate Risk")
print("1.200  -> High Risk")
print("2.420  -> Very High Risk")

print("\nBMI REFERENCE SCALE")
print("-"*45)
print("< 18.5      -> Underweight")
print("18.5-24.9   -> Healthy")
print("25-29.9     -> Overweight")
print("30+         -> Obese")

print("\nMODEL RESULTS")
print("-"*35)
print(f"Training Accuracy : {training_data_accuracy*100:.2f}%")
print(f"Testing Accuracy  : {test_data_accuracy*100:.2f}%")


# changing the input data to a numpy array

input_df = pd.DataFrame([input_data]) #type: ignore
input_data_scaled = scaler.transform(input_df)
prediction = classifier.predict(input_data_scaled)
print(prediction)

# standardize the input data (already computed as input_data_scaled)
print(input_data_scaled)

prediction = classifier.predict(input_data_scaled)
print(prediction)

print("\nPrediction Result:")
if prediction[0] == 1:
    print("🔴 The person is diabetic")
else:
    print("🟢 The person is not diabetic")