# Loan status prediction
# This is a supervised learning problem
# We are going to use SVM model which is a supervised learning model

# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd  # for data preprocessing
import seaborn as sns  # A plotting library
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Data collection and processing

# Loading the dataset to the pandas dataframe
loan_dataset = pd.read_csv("dataset.csv")

# We can check the type of the loan_dataset
print(type(loan_dataset))  # A dataframe object

# Printing the first five rows of the dataframe
print(loan_dataset.head())

# Printing the number of rows and columns
print(loan_dataset.shape)  # This is a very small dataset

# If we have a clean and a large dataset, the accuracy will be higher

# Statistical measures of the dataset
print(loan_dataset.describe())
# Only the numerical data will be shown, because the statistics cannot be calculated on the categorical data

# The number of missing values on each column
print(loan_dataset.isnull().sum())

# Here as the missing values are not large we can drop the missing value rows  ---> What we are going to do in this case
# in some cases we replace the missing value with empty string or 0
# in some cases we can do imputation (Finding the mean value and replacing that to the missing value)
# But as we have categorical data in the dataset, we cannot find the mean value of them

# dropping the missing values
loan_dataset = loan_dataset.dropna()  # dropna() drops the missing values(drops the entire row)

# The number of missing values on each column
print(loan_dataset.isnull().sum())

# label encoding
# We can replace Y and N of the labels as 1 and 0. Which will be easier for the processing
loan_dataset.replace({"Loan_Status": {'N': 0, 'Y': 1}}, inplace=True)
# In the above line {} is a dictionary data type with the keys and values
# There are several methods of doing this, In sklearn also we have a function to do this
# But in this case we use the pandas method

print(loan_dataset.head())

# Dependent column values
print(loan_dataset['Dependents'].value_counts())

# from the results we can see there is a value called 3+ which cannot be processed in our model
# So we can replace that value for any other value
# In this case we will replace all 3+ values to 4

# Replacing the value of 3+ to 4
loan_dataset = loan_dataset.replace(to_replace='3+', value=4)

# dependent values
print(loan_dataset['Dependents'].value_counts())

# Data Visualization
# The education and the loan status
sns.countplot(x='Education', hue='Loan_Status', data=loan_dataset)
plt.show()  # To show the sns count plot chart

# Marital status and Loan status
sns.countplot(x='Married', hue='Loan_Status', data=loan_dataset)
plt.show()

# We can plot and see for other attributes also if we want

# In this stage also we have several columns as text in the dataset
# Our model cannot understand text, so we need to convert them into numerical data

# Converting the categorical columns to numerical values
# Same replace procedure that we did earlier
loan_dataset.replace({'Married': {'No': 0, 'Yes': 1}, 'Gender': {'Male': 1, "Female": 0}}, inplace=True)
# We can continue the below line in the upper line, as the line gets longer, I used a separate line
loan_dataset.replace({'Self_Employed': {'No': 0, 'Yes': 1}, 'Property_Area': {'Rural': 0, "Semiurban": 1, "Urban": 2}},
                     inplace=True)
loan_dataset.replace({'Education': {'Graduate': 1, 'Not Graduate': 0}}, inplace=True)
# We use inplace=True to replace the values

print(loan_dataset.head())

# As we don't need the Loan_ID column we can drop that column

# Separating the data and label
# If we are removing multiple columns, we can include them inside a list as below
x = loan_dataset.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
y = loan_dataset['Loan_Status']

print(x)
print(y)

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=y, random_state=2)

print(x.shape, x_train.shape, x_test.shape)

# Training the model:
# Support vector machine model

# This is a classifier problem, not a Regression problem
classifier = svm.SVC(kernel='linear')  # SVC --> Support Vector Classifier

# Training the support vector machine model
classifier.fit(x_train, y_train)

# Model Evaluation
# accuracy score on training data
x_train_prediction = classifier.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)

print("The accuracy score on the training data : ", training_data_accuracy)
# 80%

# accuracy score on testing data
x_test_prediction = classifier.predict(x_test)
testing_data_accuracy = accuracy_score(x_test_prediction, y_test)

print("The accuracy score on the testing data : ", testing_data_accuracy)
# 83%

# Over-trained model means it always depends on trained data
# Which gives a higher accuracy score on train data
# and gives less accuracy score on test data
# But if both the scores are close to each other , then the model is trained good
# In this case the values are close to each other, so the model is trained good

# Making a predictive system
# --------------------------------
