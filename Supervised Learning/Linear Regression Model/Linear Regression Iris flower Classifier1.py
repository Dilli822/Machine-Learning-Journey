from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd 
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 

print("LINEAR REGRESSION VS CLASSIFICATION")
print("Classification classify the items under different class based on their similarlity")

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

# input function 
def input_fn(features, labels, training=True, batch_size=256):
    # convert the input to the datasets
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    
    # shuffle and repeat if in the training mode
    if training:
        dataset = dataset.shuffle(1000).repeat()
        
    return dataset.batch(batch_size)
    
# feature columns
# we just check/loop in with the keys and append with the myfeaturecolumns
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    
print(my_feature_columns)

# Building the Model
# Model can be build using two ways:
# 1. LinearClassifier (similar to Linear Regression)
# 2. DMClassifier (Deep Neural Network)
# 3. We can create a custom model but must follow mathematics and rules

# Builing the model with 2 hidden layers with 30 and 10 hidden nodes each
classifier = tf.estimator.DNNClassifier(
    feature_columns = my_feature_columns,
    # TWO HIDDEN LAYERS OF 30 AND 10 NODES RESPECTIVELY
    hidden_units = [30, 10],
    # the model must choose between 3 classes
    n_classes = 3
)

# training the model
# x = lamba: print("hello") lambda function is anonymous function that accepts n number of arguments
# but return only one output and here lamba is accepting the print builit in function or input_fn which is cool thing
classifier.train(
    input_fn = lambda: input_fn(train, train_y, training=True),
    steps = 5000
)


# Evaluate the model on the training data
train_eval_result = classifier.evaluate(
    input_fn=lambda: input_fn(train, train_y, training=False))

accuracy = train_eval_result['accuracy']
inaccuracy = 1 - accuracy

print("\nTraining set accuracy: {:.2f}%".format(accuracy * 100))
print("Training set inaccuracy: {:.2f}%".format(inaccuracy * 100))

# ouptut may vary
# Training set accuracy: 85.83%
# Training set inaccuracy: 14.17%

test_eval_result = classifier.evaluate(
    input_fn = lambda: input_fn(test, test_y, training=False))

accuracy_test = test_eval_result['accuracy']
inaccuracy_test = 1 - accuracy

print("\nTraining set with Test Set accuracy: {:.2f}%".format(accuracy_test * 100))
print("Training set  with Test Set inaccuracy: {:.2f}%".format(inaccuracy_test * 100))


# Define labels for the bars
labels_forPlt = ['Training', 'Testing']
colors = ['orange', 'green']
# Extract accuracy values from evaluation results
accuracy_values_plt = [train_eval_result['accuracy'], test_eval_result['accuracy']]

# Plotting the bar chart
plt.bar(labels_forPlt, accuracy_values_plt, color=colors, width=0.3)
plt.title("Bar Chart of Testing and Training Dataset Classification")
plt.ylabel("Accuracy")
plt.show()

# Sure, here are examples illustrating the use of ** for unpacking dictionaries:
# Example 1: Basic unpacking
my_dict = {'a': 1, 'b': 2, 'c': 3}
print("{a}, {b}, {c}".format(**my_dict))  # Output: 1, 2, 3

# Example 2: Using unpacked values with additional formatting
my_dict = {'name': 'Alice', 'age': 30}
print("My name is {name} and I am {age} years old.".format(**my_dict))  # Output: My name is Alice and I am 30 years old.

# Example 3: Unpacking dictionary to function parameters
def my_function(a, b, c):
    print("a:", a)
    print("b:", b)
    print("c:", c)

my_dict = {'a': 1, 'b': 2, 'c': 3}
my_function(**my_dict)  # Output:
# a: 1
# b: 2
# c: 3

# ouptut 
# feature_columns = [
#     tf.feature_column.numeric_column(key='SepalLength', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None),
#     tf.feature_column.numeric_column(key='SepalWidth', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None),
#     tf.feature_column.numeric_column(key='PetalLength', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None),
#     tf.feature_column.numeric_column(key='PetalWidth', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None)
# ]
# batch size?
# The batch size in the input_fn function you provided is set to 256 by default. 
# This means that during training (if training=True), the dataset will be divided 
# into batches, each containing 256 samples, for processing.
# In summary, the batch_size parameter controls how many examples are processed at once
# during training, impacting both computational efficiency and the dynamics of the 
# optimization process.
# pros are efficiency, grouping of data,generalization, gradient descent

