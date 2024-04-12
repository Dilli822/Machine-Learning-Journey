
# Link: https://www.tensorflow.org/tutorials/keras/regression

"""
Basic regression: Predict fuel efficiency
what is regresssion?
In statistical modeling, regression analysis is a set of statistical processes for estimating the relationships between a 
dependent variable (often called the 'outcome' or 'response' variable, or a 'label' in machine learning parlance) and one or more independent
variables (often called 'predictors', 'covariates', 'explanatory variables' or 'features'). The most common form of regression analysis is linear regression, 
in which one finds the line (or a more complex linear combination) that most closely fits the data according to a specific mathematical criterion.

Gaussian Distribution:
Gaussian distribution provides a mathematical framework for understanding how data tends to cluster around an average value with a certain degree of variation.
It's a continuous probability distribution that describes the probability of a variable occurring within a specific range.

"""


"""
Regression Problem aim is to predict the output of continous value(12.123, 45.67) like a price or proabability. We are using classic AUTO MPG, dataset and demonstrates how to build
models to predict the fuel efficiency of the later 1970s and early 1980s automobiles. To do this, we will provide the models with a description of many automobiles from that
time period (1970s - 1980s), 

Cylinders, Displacement , horsepower, and weight are the description attributes. and use we are using the keras api.

"""
# Imports 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)
# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

# The Auto MPG dataset 
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)

dataset = raw_dataset.copy()
dataset.tail()
print(dataset)

# Clean the data - clearning the dataset that contains a few unknown values
dataset.isna().sum()
dataset = dataset.dropna()  # dropping these rows to keep this initial tutorial simple
print("dropped dataset ", dataset)


dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
dataset.tail()

# As always let's split the data into training set and testing set - we will use the testing set at the final stage of evaluation

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Inspect the data --> review the joint distribution of a few pairs of columns from the training set.
# The top row suggests that the fuel efficiency (MPG)

sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
plt.show() 
# aftere training data we get the mean, std, min and total number, features from the data
train_dataset.describe().transpose()

print(train_dataset.describe().transpose())  # this will print the table with features like mean, std

# Split features from labels
# we separate the label like mean and std from the features and this label is the value that we will train the model to predict

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

# Normalization - In the table of statistics it's easy to see how different the ranges of each feature are:

train_dataset.describe().transpose()[['mean', 'std']]
# # NOTE >>> Features are mutliplied by the model weights.  xi * wi + bias where features works as input. so, the scale of the 
# outputs and the scale of the gradients are affected by the scale of the inputs. 
# so it is good practice to normalize features that use different scales and ranges. normalization makes training much more stable.
## Normalization Layer
# tf.keras.layers.Normalization is a clean and simple way to add feature normalization into our model, the first step is to create the layer
normalizer = tf.keras.layers.Normalization(axis=-1)
# Then fit the state of the preprocessing layer to the data by calling Normalization.adapt

normalizer.adapt(np.array(train_features, dtype=np.float32))
# mean and variance calculation and storing in the layer
print("normalized mean numpy -->", normalizer.mean.numpy())
# Extract the first example from 'train_features'
first_example = train_features.iloc[[0]]
# Convert the first example to a TensorFlow tensor
first_example_tensor = tf.convert_to_tensor(first_example.values, dtype=tf.float32)
# Reshape the tensor to have a shape of (1, num_features)
first_example_tensor = tf.reshape(first_example_tensor, (1, -1))
# Normalize the example using the normalization layer
normalized_example = normalizer(first_example_tensor)

print('First example:', first_example)
print()
print('Normalized:', normalized_example.numpy())


# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(train_dataset.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Histograms
train_dataset.hist(figsize=(12, 10), bins=20)
plt.suptitle('Histograms of Features')
plt.show()

# Box Plots
plt.figure(figsize=(10, 6))
sns.boxplot(x='Origin', y='MPG', data=train_dataset)
plt.title('Box Plot of MPG by Origin')
plt.show()

# Scatter Plots
sns.pairplot(train_dataset, vars=['MPG', 'Cylinders', 'Displacement', 'Weight', 'Acceleration'])
plt.show()



# Normalization is a technique used in data processing to scale the values of a dataset to a standard range.
# Suppose we have a dataset with the following exam scores:
# x normalized = std(x)/ x−mean(x)
# Math scores: 60, 70, 80, 90, 100
# English scores: 65, 75, 85, 95, 105
# To normalize this dataset, we want to scale the scores so that they have a mean of 0 and a standard deviation of 1.
# Exam scores
math_scores = np.array([60, 70, 80, 90, 100])
english_scores = np.array([65, 75, 85, 95, 105])

# Calculate mean and standard deviation
math_mean = np.mean(math_scores)
math_std = np.std(math_scores)
english_mean = np.mean(english_scores)
english_std = np.std(english_scores)

# Normalize scores
normalized_math_scores = (math_scores - math_mean) / math_std
normalized_english_scores = (english_scores - english_mean) / english_std

print("Original Math Scores:", math_scores)
print("Normalized Math Scores:", normalized_math_scores)
print("Original English Scores:", english_scores)
print("Normalized English Scores:", normalized_english_scores)



