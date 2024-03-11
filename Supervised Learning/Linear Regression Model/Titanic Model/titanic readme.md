import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.compat.v2.feature_column as fc
import pandas as pd


# tensorflow.compat.v2.feature_column is a module in TensorFlow that provides utilities for working with feature columns, which are a key component in TensorFlow's input pipeline
# for building machine learning models, particularly with structured data.
# training data - what we feed to the model for training purpose 
# testing data - data we compare with the model, test it and know how well our model is performing
# why we need to train and test the data - to know either the model have just memorized the dataset or gave different result with accuracy then testing data

# the dataset we are using have two types - categorical and numeric dataset
# there are columns that are not numeric but categorical like numeric are ages and categorical are places endinburgh

dftrain = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv") # training dataset
dfeval = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/eval.csv") # testing data
y_train = dftrain.pop("survived")
y_eval = dfeval.pop("survived")
# print(dftrain)

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
# for categorical
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()  # corrected the usage of feature_name
    # print(vocabulary)
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))  # corrected the usage of feature_name

# print(feature_columns)
# feature_columns = [
#     tf.feature_column.categorical_column_with_vocabulary_list(
#         key='sex',
#         vocabulary_list=('male', 'female'),
#         dtype=tf.string,
#         default_value=-1,
#         num_oov_buckets=0
#     ),
#     tf.feature_column.categorical_column_with_vocabulary_list(
#         key='n_siblings_spouses',
#         vocabulary_list=(1, 0, 3, 4, 2, 5, 8),
#         dtype=tf.int64,
#         default_value=-1,
#         num_oov_buckets=0
#     ),
#     tf.feature_column.categorical_column_with_vocabulary_list(
#         key='parch',
#         vocabulary_list=(0, 1, 2, 5, 3, 4),
#         dtype=tf.int64,
#         default_value=-1,
#         num_oov_buckets=0
#     ),
#     tf.feature_column.categorical_column_with_vocabulary_list(
#         key='class',
#         vocabulary_list=('Third', 'First', 'Second'),
#         dtype=tf.string,
#         default_value=-1,
#         num_oov_buckets=0
#     ),
#     tf.feature_column.categorical_column_with_vocabulary_list(
#         key='deck',
#         vocabulary_list=('unknown', 'C', 'G', 'A', 'B', 'D', 'F', 'E'),
#         dtype=tf.string,
#         default_value=-1,
#         num_oov_buckets=0
#     ),
#     tf.feature_column.categorical_column_with_vocabulary_list(
#         key='embark_town',
#         vocabulary_list=('Southampton', 'Cherbourg', 'Queenstown', 'unknown'),
#         dtype=tf.string,
#         default_value=-1,
#         num_oov_buckets=0
#     ),
#     tf.feature_column.categorical_column_with_vocabulary_list(
#         key='alone',
#         vocabulary_list=('n', 'y'),
#         dtype=tf.string,
#         default_value=-1,
#         num_oov_buckets=0
#     )
# ]


# # for numeric columns
for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))  # corrected the usage of feature_name

print("final feature columns is ", feature_columns)


# ------ output from the vocabulary -----
# [1 0 3 4 2 5 8]
# [0 1 2 5 3 4]
# ['Third' 'First' 'Second']
# ['unknown' 'C' 'G' 'A' 'B' 'D' 'F' 'E']
# ['Southampton' 'Cherbourg' 'Queenstown' 'unknown']
# ['n' 'y']

# -------- output from the feature_columns
# For the 'sex' column:
# Unique values: ['male', 'female']
# For the 'n_siblings_spouses' column:

# Unique values: [1, 0, 3, 4, 2, 5, 8]
# For the 'parch' column:

# Unique values: [0, 1, 2, 5, 3, 4]
# For the 'class' column:

# Unique values: ['Third', 'First', 'Second']
# For the 'deck' column:

# Unique values: ['unknown', 'C', 'G', 'A', 'B', 'D', 'F', 'E']
# For the 'embark_town' column:

# Unique values: ['Southampton', 'Cherbourg', 'Queenstown', 'unknown']
# For the 'alone' column:

# Unique values: ['n', 'y']
# The printed output from your loop accurately reflects the unique values present in 
# each categorical column of your DataFrame. These unique values will be used to create feature columns for your TensorFlow model.

# ----- output for second loop
# feature_columns = [
#     tf.feature_column.categorical_column_with_vocabulary_list(
#         key='sex',
#         vocabulary_list=('male', 'female'),
#         dtype=tf.string,
#         default_value=-1,
#         num_oov_buckets=0
#     ),
#     tf.feature_column.categorical_column_with_vocabulary_list(
#         key='n_siblings_spouses',
#         vocabulary_list=(1, 0, 3, 4, 2, 5, 8),
#         dtype=tf.int64,
#         default_value=-1,
#         num_oov_buckets=0
#     ),
#     tf.feature_column.categorical_column_with_vocabulary_list(
#         key='parch',
#         vocabulary_list=(0, 1, 2, 5, 3, 4),
#         dtype=tf.int64,
#         default_value=-1,
#         num_oov_buckets=0
#     ),
#     tf.feature_column.categorical_column_with_vocabulary_list(
#         key='class',
#         vocabulary_list=('Third', 'First', 'Second'),
#         dtype=tf.string,
#         default_value=-1,
#         num_oov_buckets=0
#     ),
#     tf.feature_column.categorical_column_with_vocabulary_list(
#         key='deck',
#         vocabulary_list=('unknown', 'C', 'G', 'A', 'B', 'D', 'F', 'E'),
#         dtype=tf.string,
#         default_value=-1,
#         num_oov_buckets=0
#     ),
#     tf.feature_column.categorical_column_with_vocabulary_list(
#         key='embark_town',
#         vocabulary_list=('Southampton', 'Cherbourg', 'Queenstown', 'unknown'),
#         dtype=tf.string,
#         default_value=-1,
#         num_oov_buckets=0
#     ),
#     tf.feature_column.categorical_column_with_vocabulary_list(
#         key='alone',
#         vocabulary_list=('n', 'y'),
#         dtype=tf.string,
#         default_value=-1,
#         num_oov_buckets=0
#     ),
#     tf.feature_column.numeric_column(
#         key='age',
#         shape=(1,),
#         default_value=None,
#         dtype=tf.float32,
#         normalizer_fn=None
#     ),
#     tf.feature_column.numeric_column(
#         key='fare',
#         shape=(1,),
#         default_value=None,
#         dtype=tf.float32,
#         normalizer_fn=None
#     )
# ]



# tensorflow: This is the main TensorFlow library, which is widely used for building machine learning models.

# compat: This stands for "compatibility". The compat module in TensorFlow provides compatibility utilities for using TensorFlow 1.x-style code in TensorFlow 2.x. It allows you to use certain functionalities from TensorFlow 1.x in TensorFlow 2.x environments.

# v2: This signifies TensorFlow version 2.x, which introduced significant changes and improvements over version 1.x. TensorFlow 2.x emphasizes ease of use, eager execution by default, and improved integration with Python.

# feature_column: This is the specific module within TensorFlow that deals with feature columns. Feature columns are used to represent and transform 
# raw input features into a format that can be fed into TensorFlow models. They handle various types of input data, such as categorical data 
# (e.g., strings or integers representing categories) and numerical data (e.g., floating-point numbers).


# In summary, tensorflow.compat.v2.feature_column provides utilities for working with 
# feature columns in TensorFlow 2.x environments while maintaining compatibility with 
# TensorFlow 1.x-style code. It enables you to handle structured data effectively for training machine learning models.

