# Importing the libraries
import matplotlib.pyplot as plt  # This library is used for plotting
import pandas as pd
import seaborn as sns  # This library is also used for plotting
from sklearn import metrics  # for evaluating our model
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor  # Importing the xgboost regression algorithm

# loading the Boston House Price Dataset
house_price_dataset = pd.read_csv("Boston.csv")

# Printing the first five rows of the dataset
print(house_price_dataset.head())

# Checking the number of rows and columns in the dataset
print(house_price_dataset.shape)

# Check for missing values
# This will give the sum of missing values in each column
print(house_price_dataset.isnull().sum())
# So we found that no missing values in this dataset(if found, we have to do more processing to remove them)

# Statistical measures of the dataset
# This is only to understand our dataset better
print(house_price_dataset.describe())

# Understanding the correlation between various features in the dataset
# Correlation basically represents the relationship between two variables
# Two types of correlation
# 1 ---> Positive correlation
# 2 ---> Negative correlation

correlation = house_price_dataset.corr()  # This will find the correlation between all the values

# Constructing a heatmap to understand the correlation
plt.figure(figsize=(10, 10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size': 8}, cmap='Blues')
plt.show()

# cbar=True means we want the color bar in the side
# square=True means we want all the values to be inside a square
# fmt ='.1f' means how many float values that we want (1f means we want 1 value after the decimal point)
# annot means the annotations (Feature names and all the values)
# annot_kws={'size:'} we can set the size of the annotations
# cmap is the color of the mapping

# Splitting the data and the labels
x = house_price_dataset.drop(["price"], axis=1)
y = house_price_dataset["price"]

print(x)
print(y)

# Splitting the data into training data and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

print(x.shape, x_train.shape, x_test.shape)

# Model training
# XGBoost Regressor model (type of Decision tree based model), (2 or more models together)

# Loading the model
model = XGBRegressor()   # This will load the XGBRegressor model to the model variable

# training the model with x_train
model.fit(x_train, y_train)

# Evaluation
# We cannot find the accuracy score as the previous classification models as this is a regression model
# Because all the labels are numerical values. Not boolean or symbolic values
# Therefore we use the r square error and mean deviation error

# Prediction on training data
# accuracy for prediction on training data
training_data_prediction = model.predict(x_train)
print(training_data_prediction)

# R squared error
score_1 = metrics.r2_score(y_train, training_data_prediction)

# Mean Absolute error
score_2 = metrics.mean_absolute_error(y_train, training_data_prediction)

print("R squared error : ", score_1)  # If the value is close to 1, then it is good
# if 0, then our model is performing perfectly (Lesser the value, more accurate)

print("Mean Absolute error value : ", score_2)

# Visualizing the actual prices and predicted prices
plt.scatter(y_train, training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual price vs Predicted price")
plt.show()

# Prediction on testing data
# accuracy for prediction on testing data
testing_data_prediction = model.predict(x_test)
print(testing_data_prediction)

# R squared error
score_1 = metrics.r2_score(y_test, testing_data_prediction)

# Mean Absolute error
score_2 = metrics.mean_absolute_error(y_test, testing_data_prediction)

print("R squared error : ", score_1)  # If the value is close to 1, then it is good
# if 0, then our model is performing perfectly (Lesser the value, more accurate)

print("Mean Absolute error value : ", score_2)
