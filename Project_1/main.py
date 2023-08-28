# Importing the libraries
import numpy as np                                          # for creating numpy arrays
import pandas as pd                                         # for loading our data into data tables (data points)
from sklearn.model_selection import train_test_split        # to automatically split the train and test data
from sklearn.linear_model import LogisticRegression         # to solve classification problems
from sklearn.metrics import accuracy_score                  # for calculating the accuracy of the output result

# Data Collection and Data Processing
# Loading the dataset to the pandas data frame

sonar_data = pd.read_csv("sonarData.csv", header=None)  # As there are no headings for the columns of the csv file

print(sonar_data.head())  # Prints the first five rows of data

# Number of rows and columns
print(sonar_data.shape)

# Prints the statical measures of the data
print(sonar_data.describe())

# Counts how many rocks and mines
print(sonar_data[60].value_counts())  # inside the square brackets we have the column number of the counting column

# Normally this number of data records are not enough for a very good accurate model
# But here we are just testing some example (More data ---> More accuracy of the model)

# Grouping the data based on mine and rock and finding the mean value for each column
print(sonar_data.groupby(60).mean())

# Separating the data and the labels
# Numerical values are the data
# Last column data are the labels

# Putting all the data in the variable x by dropping the last column
x = sonar_data.drop(columns=60, axis=-1)  # if dropping a column, axis is -1 # if dropping a row, axis is 0d

# Putting the labels in the variable y
y = sonar_data[60]

print(x)
print(y)

# Splitting the data into training and test data
# The below variable order should be followed
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=y, random_state=1)

# x_train --> the training data
# x_test --> testing data

# y_train --> label of the training data
# y_test --> label of the test data

# parameters of the above method
# x, y means we are going to split x and y into training and test data
# test_size = 0.1 means we need 10% of the data as test data

print(x.shape, x_train.shape, x_test.shape)

# Printing the training data
print(x_train, y_train)

# Training the machine learning model with x_train data
# we are using the logistic regression model

# this will load the logistic regression to the variable model
model = LogisticRegression()

# Training the model with training data
model.fit(x_train, y_train)

# Model Evaluation
# Finding the accuracy on the training data
# Any accuracy more than 70% is good

x_train_prediction = model.predict(x_train)

# x_train_prediction is the prediction of the values of training data
# y_train is the actual answer of the prediction
training_data_accuracy = accuracy_score(x_train_prediction, y_train)

print("The accuracy on the training data : ", training_data_accuracy)
# so we get 0.83... accuracy which means about 83% accuracy which is good

# Finding the accuracy on the test data
x_test_prediction = model.predict(x_test)

# x_test_prediction is the prediction of the values of test data
# y_test is the actual answer of the prediction
test_data_accuracy = accuracy_score(x_test_prediction, y_test)

print("The accuracy on the test data : ", test_data_accuracy)
# so we get 0.76... accuracy which means about 76% accuracy which is good

# ------------- Making a prediction system
# copies a record from the data without the answer
# opened the Excel sheet using the notepad add automatically the commas
# The below data should give the output as a rock
input_data = (0.0270, 0.0092, 0.0145, 0.0278, 0.0412, 0.0757, 0.1026, 0.1138, 0.0794, 0.1520, 0.1675, 0.1370, 0.1361, 0.1345, 0.2144, 0.5354, 0.6830, 0.5600, 0.3093, 0.3226, 0.4430, 0.5573, 0.5782, 0.6173, 0.8132, 0.9819, 0.9823, 0.9166, 0.7423, 0.7736, 0.8473, 0.7352, 0.6671, 0.6083, 0.6239, 0.5972, 0.5715, 0.5242, 0.2924, 0.1536, 0.2003, 0.2031, 0.2207, 0.1778, 0.1353, 0.1373, 0.0749, 0.0472, 0.0325, 0.0179, 0.0045, 0.0084, 0.0010, 0.0018, 0.0068, 0.0039, 0.0120, 0.0132, 0.0070, 0.0088)

# changing the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshaping the numpy array as we are predicting for a one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# 1, -1 is we are predicting for a one instance, and we are predicting the label for one instance
prediction = model.predict(input_data_reshaped)
print(prediction)

# It predicted the output as a Rock, so it's a correct prediction

# let's give the output as rock or a mine
# As the output of the prediction is given inside a list, we will use the index 0

if prediction[0] == "R":
    print("The object is a Rock")
else:
    print("The object is a mine")





