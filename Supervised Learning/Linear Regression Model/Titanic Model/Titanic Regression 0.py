# import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
# import tensorflow.compat.v2.feature_column as fc
# import pandas as pd


# dftrain = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv") # training dataset
# dfeval = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/eval.csv") # testing data
# y_train = dftrain.pop("survived")
# y_eval = dfeval.pop("survived")
# # print(dftrain)

# CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
# NUMERIC_COLUMNS = ['age', 'fare']

# feature_columns = []
# # for categorical
# for feature_name in CATEGORICAL_COLUMNS:
#     vocabulary = dftrain[feature_name].unique()  # corrected the usage of feature_name
#     # print(vocabulary)
#     feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))  # corrected the usage of feature_name


# for feature_name in NUMERIC_COLUMNS:
#     feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))  # corrected the usage of feature_name

# print("final feature columns is ", feature_columns)


# # training process
# # 1. first we need dataset
# # 2. since we are training the machine dataset will not be small but larger dataset
# # 3. sometimes it could be terabytes size so there are no such RAMs in computer that can load up the tb datasets
# # 4. so we use technique called ephocs where big size dataset is broken down or only portion of data is
# # 5. fed into a machine for n times with small batches of entries
# # 6. so if we have 10 ephocs, our model will see the same dataset 10 times
# # 7. epochs is a simply one stream of our entire dataset
# # INPUT FUNCTION - Since we need to fed the dataset into the model and we are using the panda data set 
# # so we need to convert the panda dataset into an object.
# # since tensorflow dataset comes into an object that means panda dataset ---> parse into ---> tensor data set object


# def make_input_fn(data_df, label_df, num_epochs=10, shuffle = True, batch_size = 32):
#     def input_function():
#         ds = tf.data.Dataset.from_tensor_slices(dict(data_df), label_df)
#         # create tf.data.Dataset object with data and its dict and label
#         if shuffle:
#             ds = ds.shuffle(1000) # randomize the order of data
#         ds = ds.batch(batch_size).repeat(num_epochs) 
#         # split the dataset into a batches of 32 and repeat process for number of epochs
#         return ds # return a batch of dataset 
    
#     return input_function # return the object  function for the use

# train_input_fn = make_input_fn(dftrain, y_train)
# eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)


# # passing LinearClassifier object from the estimator model of the tensorflow which creates
# # a model for us that means estimator model in the tensorflow has linearclassifier model
# # that is accepting the feature_columns = feature_columns

# linear_est = tf.estimator.LinearClassifier(feature_columns= feature_columns)

# # ----------- Now training the model ------------
# linear_est.train(train_input_fn)  # train the given param which is input function itself
# result = linear_est.evaluate(eval_input_fn) # get model metrics/stats by testing on testing data

# clear_output() # clears console
# print(result["accuracy"]) # the result variable is simply a dict of stats about our model



import numpy as np
import tensorflow as tf
import pandas as pd

# Load data
dftrain = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv") # training dataset
dfeval = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/eval.csv") # testing data
y_train = dftrain.pop("survived")
y_eval = dfeval.pop("survived")

# Define feature columns
feature_columns = []
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

# Define input function
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function

train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

# Define and train the model
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)

# Print accuracy
print(result['accuracy'])
# we get output accuracy at 0.7462121 
print(result)
# GIVES OUTPUT IN THE OBJECT FORMAT REMEMBER WE HAVE DONE PARSE IN OBJECTS
# {
#     'accuracy': 0.74242425,
#     'accuracy_baseline': 0.625,
#     'auc': 0.8375268,
#     'auc_precision_recall': 0.7862294,
#     'average_loss': 0.47628114,
#     'label/mean': 0.375,
#     'loss': 0.4689349,
#     'precision': 0.65346533,
#     'prediction/mean': 0.37340146,
#     'recall': 0.6666667,
#     'global_step': 200
# }
# SO IF CHANGE THE EPOCHS THE ACCURACY WILL ALSO VARY AS WE ARE SHUFFLING THE DATA SET HERE

