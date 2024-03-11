#Keras is an open-source deep learning library written in Python. It is designed to be user-friendly, modular, and extensible, allowing for easy and fast prototyping of neural network models. Keras was developed with a focus on enabling rapid experimentation and is built on top of other libraries such as TensorFlow, Theano, or Microsoft Cognitive Toolkit (CNTK)
# By using these __future__ imports, you can write code that is more compatible with Python 3 while still being able to run on Python 2. This can help make the transition to Python 3 smoother and reduce the amount of code that needs to be modified when porting projects..
from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd 
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 

print("LINEAR REGRESSION VS CLASSIFICATION")
print("Classification classify the items under different class based on their similarlity")
# instead of numeric values it classify the items under the class or categories
# separate data points into classes of different labels


# ------ we are given dataset and variety of the flowers into 3 different species-----
# Setosa
# Versicolor
# Virginica

# ------ given information of each flower is ---
# sepal length
# sepal width
# petal length
# petal width


# using keras a module instead of tensorflow to grab our datasets and read them into pandas dataframes
CSV_COLUMN_NAME = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
    
)

test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"
)

# row 0 is a header row
train = pd.read_csv(train_path, names=CSV_COLUMN_NAME, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAME, header=0)

train.head()
print("train data -->")
print(train.head())

print(test.columns)

# now we pop the species for training and testing purpose
train_y = train.pop('Species')
test_y = test.pop('Species')

# The label column has now been removed from the features
print(train.head())

# printing the shape
print(train.shape)   # (120, 4) in 4 columns 120 entries

# numpy array spliting is the reverse of joining
arr = np.array([2, 4, 6, 8])
newarr = np.array_split(arr, 3)
print("splited array is ", newarr)
# splited array is  [array([2, 4]), array([6]), array([8])]
# if array has less than number to be splitted then it will adjust accordingly
print("array_split will auto adjust numbers to be split ")

adjust_arr = np.array_split(arr, 5)
print(adjust_arr)
# [array([2]), array([4]), array([6]), array([8]), array([], dtype=int64)]

x = np.array(["A", "B", "C", "D"])
y = np.array([3, 8, 1, 10])
plt.barh(x, y, color="#4CAF50")
plt.show()

plt.title("with width = 0.5")
plt.bar(x, y, width=0.5)
plt.show()

plt.title("with height for horizontal bars = 0.6 and color pink")
plt.barh(x,y, height=0.2, color="pink")
plt.show()