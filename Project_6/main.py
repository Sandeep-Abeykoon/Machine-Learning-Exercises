# Wine Quality Prediction
# We are going to use Random Forest model
# Random Forest is one of the most important supervised learning model in ML

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # To make graphs
import seaborn as sns  # Important in data visualization
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Data collection
# Loading the dataset to a pandas dataframe
wine_dataset = pd.read_csv("winequality-red.csv")

# Number of rows and columns in the dataset
print(wine_dataset.shape)

# The first five rows of the dataset
print(wine_dataset.head())

# Checking for missing values
print(wine_dataset.isnull().sum())
# No missing values

# Data Analysis and Visualization
# Statistical measures of the data set
print(wine_dataset.describe())

# Number of values for each quality value
sns.catplot(x='quality', data=wine_dataset, kind='count')
plt.show()  # Showing the catplot chart

# volatile_acidity vs quality
plot = plt.figure(figsize=(5, 5))
sns.barplot(x='quality', y='volatile acidity', data=wine_dataset)
plt.show()

# citric acid vs quality
plot = plt.figure(figsize=(5, 5))
sns.barplot(x='quality', y='citric acid', data=wine_dataset)
plt.show()

# Correlation
# Positive Correlation
# Negative correlation --> Inverse proportion
correlation = wine_dataset.corr()

# Constructing a heatmap to understand the correlation between the columns
plt.figure(figsize=(10, 10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size': 8}, cmap='Blues')
plt.show()

# Data Preprocessing
# Separating the data anf the labels
x = wine_dataset.drop('quality', axis=1)
print(x)

# Label Binarization (Transforming the quality into two values)
# If the quality is greater than 7, then it's good. Else bad

y = wine_dataset['quality'].apply(lambda y_value: 1 if y_value >= 7 else 0)
# We can use the apply function to change the values using a condition as above
# In the above line we have converted to 1 and 0 because it's easier to process in the model
# This is called as label binarization or label encoding
print(y)

# Training and test data splitting process
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

print(y.shape, y_train.shape, y_test.shape)

# Model Training
# Random Forest Classifier model --> Uses more than one model in combination
# Single Decision tree contains only one decision tree
# Random Forest has n number of decision trees (More number of d.trees --> More accurate is the result)

model = RandomForestClassifier()  # Loading the RFC into the model variable
model.fit(x_train, y_train)

# Model Evaluation
# Accuracy Score
# Accuracy on the training data
x_train_prediction = model.predict(x_train)
train_data_accuracy = accuracy_score(x_train_prediction, y_train)
print("The training data accuracy : ", train_data_accuracy)

# Accuracy on the test data
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)
print("The testing data accuracy : ", test_data_accuracy)

# Building a predictive system
# This should give the output as good quality
input_data = (7.3, 0.65, 0.0, 1.2, 0.065, 15.0, 21.0, 0.9946, 3.39, 0.47, 10.0)

# Changing the input data tuple to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshaping the data as we are predicting the label for only one instance
input_data_reshape = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshape)
print(prediction)

if prediction[0] == 1:
    print("Good Quality wine")
else:
    print("Bad quality wine")
