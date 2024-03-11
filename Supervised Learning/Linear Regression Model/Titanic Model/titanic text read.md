from __future__ import absolute_import, division, print_function, unicode_literals
from IPython.display import clear_output
from six.moves import urllib

# Other import statements and code follow here

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# For Linear Regression 
# We are going to the use the dataset and find the titanic survival and death rate
# we use .csv file --> comma separated values is a text file format that uses comma to 
# separate the values and newlines to separate records

# data - we need to understand the data since data is a like a fuel to run a machine
# survived ,  gender, age, n_siblings, fare, class, deck

# we have two types of dataset for now -- training and testing dataset
# training dataset is used for training model purpose
# testing data is a separate dataset for testing the output of the trainded model
# since model can give biased output so we must use another fresh dataset

dftrain = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv") # training dataset
dfeval = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/eval.csv") # testing data
dftrain.head()  # reads the dftrain csv file

print(dftrain.head())
 
y_train = dftrain.pop("survived")
y_eval = dfeval.pop("survived")

# pd.read_csv() will return to us a new pandas dataframe. dataframe is nothing but a table
# for example we have printed 
# print(dftrain.head() )
#    survived     sex   age  n_siblings_spouses  ...  class     deck  embark_town alone
# 0         0    male  22.0                   1  ...  Third  unknown  Southampton     n
# 1         1  female  38.0                   1  ...  First        C    Cherbourg     n
# 2         1  female  26.0                   0  ...  Third  unknown  Southampton     y
# 3         1  female  35.0                   1  ...  First        C  Southampton     n
# 4         0    male  28.0                   0  ...  Third  unknown   Queenstown     y

print("trained survival ",  y_train.head()) # output of trained dataset 
# trained 
# 0    0
# 1    1
# 2    1
# 3    1
# 4    0
# Name: survived, dtype: int64

print("----")
print("testing survival ", y_eval.head())

# printing the specific
print("specific data of the training dataset", dftrain.loc[0])

print("---------")
# printing the specific
print("specific data of the training dataset", dfeval.loc[0])

# describe the dataset
print("describing the dataset ", dftrain.describe())

# shape
print("shape of the dataset ")
print(dftrain.shape)

# lets visualize the dataset since   y_train.head() and  y_eval.head() are giving us
# 0s and 1s only we must visualize using any form of graph 


plt.hist(dftrain['age'], bins=20)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Histogram of Age')
plt.show()

# higher the historgram height it is the spot where output is concentrated
# there are outliers like in this hist we have 70 and 80

# finding the males and females number
# This code plots the count of each gender in the 
# sex column of dftrain as a horizontal bar plot. It looks correct.
# dftrain.sex.value_count().plot(kind='barh')

# # This code plots the count of each class in the
# # 'class' column of dftrain as a horizontal bar plot. It seems fine.
# dftrain['class'].value_counts().plot(kind='barh')

# # This code calculates the mean survival rate grouped by gender and plots it as a horizontal bar
# # plot. It then sets the x-label to "% survive". It appears correct.
# pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel("% survive")



# Plot the count of males and females
dftrain['sex'].value_counts().plot(kind='barh')
plt.xlabel('Count')
plt.ylabel('Sex')
plt.title('Count of Males and Females')
plt.show()

# Plot the count of classes
dftrain['class'].value_counts().plot(kind='barh')
plt.xlabel('Count')
plt.ylabel('Class')
plt.title('Count of Classes')
plt.show()

# Plot the percentage of survival by sex
survival_by_sex = pd.concat([dftrain, y_train], axis=1).groupby('sex')['survived'].mean()
survival_by_sex.plot(kind='barh')
plt.xlabel('% Survived')
plt.ylabel('Sex')
plt.title('Percentage of Survived by Sex')
plt.show()

# output
# 1. The majority  of the passenger are their in 20s and 30s
# 2. Majority of the passengers are male
# 3. Majority of the people are third class passenger
# 4. Female have higher percentage of survival than males