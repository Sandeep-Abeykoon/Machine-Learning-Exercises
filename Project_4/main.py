# Fake News Prediction
# Here we are dealing with textual data

# About the dataset
# 1 ---> id: unique id for a news article
# 2 ---> title: title of a news article
# 3 ---> author: author of the news article
# 4 ---> text: the text of the article; could be incomplete
# 5 ---> label: a label that marks whether the news article is real or fake

# 1: Fake news
# 0: real news

# Importing the libraries
import pandas as pd
import numpy as np
import re  # Important to searching the text in a document
from nltk.corpus import stopwords  # Natural language toolkit, to remove the unnecessary words which has no value
from nltk.stem.porter import PorterStemmer  # does stem and returns the root word of a particular word
from sklearn.feature_extraction.text import TfidfVectorizer  # Used to convert the text into feature vectors (Numbers)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import nltk

nltk.download('stopwords')  # Downloading the stopwords from the nltk package

# Printing the stopwords in english
print(stopwords.words('english'))
# Stopwords are the words that do not carry any value to our processing (we have to remove those words)

# Data preprocessing
# Loading the dataset to a pandas dataframe
news_dataset = pd.read_csv("data.csv")

# Checking the number of rows and columns of the dataset
print(news_dataset.shape)  # (rows, columns)

# Printing the first five rows of the dataframe
print(news_dataset.head())

# Counting the number of missing values in the dataset
print(news_dataset.isnull().sum())

# id --> 0
# title --> 558
# author --> 1957
# text --> 39
# label --> 0

# we can either drop the missing values or replace the missing values with the null string
# As we have a very large dataset we can replace the missing values with a null string
# If the data set is very small then if we drop or replace null string it will affect the accuracy of the model
# But in this case, the dataset is large, so no big issue

# Replacing the null values with empty string
news_dataset = news_dataset.fillna('')  # Filling the missing values with a null string

# Merging the author name and news title
# We are doing this because we are not considering the text in this project. because it needs a lot of work
# We will be using only the author and the title for prediction in this project
news_dataset['content'] = news_dataset['author'] + " " + news_dataset['title']
# in above line, creating a new column called content in the dataframe and storing the content

# Printing the new column
print(news_dataset['content'])
# We will be using the content data and the labels to do the predictions

# Separating the data and the labels
x = news_dataset.drop(columns="label", axis=1)  # removing the label column
y = news_dataset['label']  # only storing the labels in the variable

print(x)
print(y)

# Stemming:
# This is the process of reducing a word to it's root word
# Ex- actor, actress, acting ---> act (root word)

port_stem = PorterStemmer()  # Loading the function to the variable


# Creating a function (As we might need to repeat the process multiple times)
def stemming(content):  # Line 1
    stemmed_content = re.sub("[^a-zA-Z]", " ", content)  # Line 2
    stemmed_content = stemmed_content.lower()  # Line 3
    stemmed_content = stemmed_content.split()  # Line 4
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]  # 5
    stemmed_content = ' '.join(stemmed_content)  # Line 6
    return stemmed_content  # Returning the processed content


# Line 1 --> Content is the input text

# Line 2 --> the sub() function substitutes certain values
# Line 2 --> we are excluding everything that is not present in the Regex pattern (Everything which is not alphabets)
# Line 2 --> from the content
# Line 2 --> ' ' is all the exclusions will be replaced by a space

# Line 3 --> Converting all the content to lowercase

# Line 4 --> The content will be split from spaces and converted into a list

# Line 5 --> Stemming for each word and getting rid of the stopwords

# Line 6 --> Joining all the words from the whitespace

# Applying the function to the content
news_dataset['content'] = news_dataset['content'].apply(stemming)
# The above line applies the stemming function for all the cells of the content
# and replaces the content with processed content

print(news_dataset['content'])

# If we want we also can use the text column, but it will need huge processing and time

# Separating the data and label
x = news_dataset['content'].values
y = news_dataset['label'].values

print(x)
print(y)
print(y.shape)

# Still the value is in textual form
# The computers will not understand text
# So we need to convert the text into meaningful values

# Converting the textual data to numerical data
vectorizer = TfidfVectorizer()  # This counts the number of times a particular word written in a document
# So the repetition of a word gives the value how important is that word
# And also finds the repeating non-important words and reduces the important value

vectorizer.fit(x)  # fitting our content to the vectorizer
# We don't need to do this to y because y is already a number

x = vectorizer.transform(x)  # Will convert all the values into their respective features
print(x)  # Now this will print the feature vectors according to the importance

# If we feed textual data to our ML model it cannot understand them
# But now we can feed our vectorized data to the ML model

# Splitting the dataset to training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=2)
# If you don't mention stratify = y the real news and the fake news will not be divided in an equal proportion
# If you put stratify = y it will be divided in the original proportion of the original dataset y column
# to divide the training and testing data in an equal proportion of labels T and F
# random state is used to divide the data in random form

# Training the Model : Logistic Regression model
model = LogisticRegression()
model.fit(x_train, y_train)  # Training

# Evaluation
# Accuracy score

# Accuracy score on the training data
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)

print("Accuracy score of the training data : ", training_data_accuracy)

# The accuracy of the above is 0.986 which means almost 99% which is perfect
# this is because we used a very large dataset
# For binary classification problems, logistic regression is the best model

# But the accuracy score of the testing data is the most important

# Accuracy score on the testing data
x_test_prediction = model.predict(x_test)
testing_data_accuracy = accuracy_score(x_test_prediction, y_test)

print("Accuracy score of the testing data : ", testing_data_accuracy)

# The accuracy score is 0.979 which is almost 98% which is perfect

# Making a predictive system
x_new = x_test[0]  # The fist row of the x_test column

prediction = model.predict(x_new)
print(prediction)

if prediction[0] ==0:
    print("The news is real")
else:
    print("The news is fake")

