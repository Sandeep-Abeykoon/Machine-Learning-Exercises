# Importing the libraries
import numpy as np  # to work with the numpy arrays
import pandas as pd  # to structure our data
from sklearn.preprocessing import StandardScaler  # StandardScaler is used to standardize the data to a common range
from sklearn.model_selection import train_test_split  # to split the data into training data and test data
from sklearn import svm  # Importing the support vector machine
from sklearn.metrics import accuracy_score  # to test the accuracy of the output prediction data

# Data collection and analysis
# PIMA Diabetes Dataset (Contains the diabetes data of Women)

# Loading the diabetes dataset to a pandas dataframe
diabetes_dataset = pd.read_csv("diabetes.csv")

# Printing the first five rows of the dataset
print(diabetes_dataset.head())

# Number of rows and columns of the dataset
print(diabetes_dataset.shape)

# Getting the statistical measures of the data
print(diabetes_dataset.describe())

# Checking the outcome column for number of diabetic and non-diabetic patients
# ["Outcome"] is the column that we are counting
print(diabetes_dataset["Outcome"].value_counts())

# Normally a good dataset will have thousands or lacks of data

# Calculating the mean value of the diabetic and non-diabetic groups
# We can see the important stats of the mean value by referring the mean data
print(diabetes_dataset.groupby("Outcome").mean())

# Separating the data and the outcome
# axis=-1 if dropping a column
# axis=0 if dropping a row

x = diabetes_dataset.drop(columns="Outcome", axis=-1)  # Dropping the last column
y = diabetes_dataset["Outcome"]  # Only the outcome data is here

print(x)
print(y)

# Data Standardization (One of the important steps in data preprocessing)
# Standardization is like organizing the data to an acceptable range omitting random peaks and lows
# Which will be easier for the machine learning model to be trained

scaler = StandardScaler()
scaler.fit(x)  # We are shifting all the inconsistent data using the StandardScaler() Function

standardized_data = scaler.transform(x)  # Transforming all the data to a common range

# Without using two functions to fit and transform, we can use fit_transform() function to do both the things at once

print(standardized_data)
# When we see the output, we can see that all the values are 0. and 1. So our ML model can easily identify the values

x = standardized_data
y = diabetes_dataset["Outcome"]

print(x)
print(y)

# Train Test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)

# x = data
# y = labels
# test_size --> Testing data splitting size from the entire data (0.2 means 20% of the data)

# stratify --> Sampling the data for the training data and testing data
# according to the same proportion of the y column in the same proportion as the original data set

# random_state --> to randomize the splitting of the data

print(x.shape, x_train.shape, x_test.shape)

# Training the model
classifier = svm.SVC(kernel='linear')

# Training the support vector machine classifier
classifier.fit(x_train, y_train)

# Model Evaluation
# Accuracy Score on the training data
# this will predict all the labels for the x_train data
x_train_prediction = classifier.predict(x_train)

# comparing the predicted labels with actual labels
training_data_accuracy = accuracy_score(x_train_prediction, y_train)

print("Accuracy score of the training data : ", training_data_accuracy)
# we get the accuracy score of 0.78... which is about 78%, we also can use various optimized techniques to increase this

# Accuracy Score on the testing data
# this will predict all the labels for the x_test data
x_test_prediction = classifier.predict(x_test)

# comparing the predicted labels with actual labels
testing_data_accuracy = accuracy_score(x_test_prediction, y_test)

print("Accuracy score of the testing data : ", testing_data_accuracy)
# we get the accuracy score of 0.77... which is about 77%, we also can use various optimized techniques to increase this

# Making a predictive system
# Inputting a data record of a person who has diabetics (Except the last column/data label)
input_data = (1, 189, 60, 23, 846, 30.1, 0.398, 59)

# Changing the input tuple data into a numpy array
# Processing a numpy array is easier and efficient

input_data_as_numpy_array = np.asarray(input_data)

# Reshape the array as we are predicting for one instance
# If we don't reshape, then our model expects 765 (Number of records of the training data) data points
# We are going to tell the model that we need the prediction only for a one data point
# Reshaping the data using a numpy array is very easy

input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
# 1 ---> we are trying to predict only one instance

# But now there is a problem, we can't directly give the above data to predict
# Why? because we have standardized the data while training the model (We haven't used the raw data)

# Now we have to do the same procedure here
# Standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

# Prediction
prediction = classifier.predict(std_data)
print(prediction)

if prediction[0] == 1:
    print("The person has diabetes")
else:
    print("The person has no diabetic")
