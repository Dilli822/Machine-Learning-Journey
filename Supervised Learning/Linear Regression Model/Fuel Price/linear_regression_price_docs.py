
# Link: https://www.tensorflow.org/tutorials/keras/regression

"""
Basic regression: Predict fuel efficiency
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

# sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
# plt.show() 
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

# Normalization is a technique used in data processing to scale the values of a dataset to a standard range.
# Before diving into a deep neural network model, we start with linear regression using one and several variables

"""
Linear Regression with One Variable:
- Let's use a single variable linear regression to predtic 'MPG' from 'HorsePower'
- we start traning a model with tf.keras typically by defining the model architecture using tf.keras.Sequential model which represents a sequence of steps.

There are two steps in our single variable linear regression model:
 - Normalze the 'horspower' input features using the tf.keras.layers.Normalization preprocessing layer.
 - Apply a linear transformation y = mx + b to product 1 output using a linear layer(tf.keras.layers.Dense)
 
the number of inputs can be either be set by the input_shape argument, or automatically when the model is run for the first time.

"""

# first NUmPY array made of the 'HorsePower' features 
horsepower = np.array(train_features['Horsepower'])

# and instantitate the tf.keras.layers.Normalization and fit its state to the horsepower data.
horsepower_normalizer = layers.Normalization(input_shape=[1,], axis = None)
horsepower_normalizer.adapt(horsepower)


# Building keras sequential model

horsepower_model = tf.keras.Sequential([
    horsepower_normalizer,
    layers.Dense(units=1)
])
horsepower_model.summary()

print(horsepower_model.summary())

# This model will predict 'MPG' from 'HorsePower' -  runining on the untrained model on the first 10 horsepower values the output wont be good but notice that it has the expected shape of(10,1)
firstbuild = horsepower_model.predict(horsepower[:10])
print(firstbuild)

# Once the model is built, configure the training procedure using the keras Model.Compile method, while compiling model the important arguments are loss and optimizer, since these define what will
# be optimized since these define what will be optimized (mean_absolute_error) and how()using the tf.keras.optimizer adam

# Adam is used to minimize the loss function during the training of neural networks.
# Adam can be looked at as a combination of RMSprop and Stochastic Gradient Descent with momentum.
# Gradient descent is an iterative optimization algorithm for finding the local minimum of a function.
# The loss function is a method of evaluating how well your machine learning algorithm models your featured data set.
# RMSprop (Root Mean Squared Propagation) is an optimization algorithm , Useful for preventing underflow during mixed precision training. 

horsepower_model.compile(
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate = 0.1),
    loss="mean_absolute_error"
)

# lets fit the model to execute the training for 100 epochs

history = horsepower_model.fit(
    train_features['Horsepower'],
    train_labels,
    epochs=100,
    # suppress logging,
    verbose = 0,
    # calculate  validation results on 20% of the training data
    validation_split = 0.2
)

# lets visualze the model's training progress using the stats stored in the history object

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG] ')
    plt.legend()
    plt.grid(True)

plot_loss(history)


# Collecting the results on the test set for later:
test_results = {}

test_results['horsepower_model'] = horsepower_model.evaluate(
    test_features['Horsepower'],
    test_labels, verbose=0)

# lets view the model's predictions as a function of the input
x = tf.linspace(0.0, 250, 251)
y = horsepower_model.predict(x)

# Assuming 'test_features' and 'test_labels' are your test dataset
test_predictions = horsepower_model.predict(test_features['Horsepower']).flatten()

def plot_horsepower(x, y):
    plt.figure()  # Create a new figure
    plt.scatter(test_features['Horsepower'], test_labels, label='Data', color='blue')  # Actual data points
    plt.plot(x, y, color='red', label='Predictions')  # Predicted values
    plt.xlabel('Horsepower')
    plt.ylabel('MPG')
    plt.title('Horsepower vs MPG')
    plt.legend()
    plt.show()

# Generate x values from 0 to 250 for plotting
x = tf.linspace(0.0, 250, 251)
# Predict y values based on the x values
y = horsepower_model.predict(x)

# Plot the data and predictions
plot_horsepower(x, y)


# Regression with DNN

"""
Code is same except the model is expanded to include some "hidden" non-linear layers, the hidden here just means not directly connected to the inputs or outputs
These models will contain a few more layers than the linear model:
 - Normalization layer, as before(with horsepower_normalizer) for a single input model
 - Two Hideen, non-linear, Dense Layers with ReLu activation function non-linearity
 - all linear dense single-output layer

"""

def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    
    model.compile( loss='mean_absolute_error', optimizer = tf.keras.optimizers.legacy.Adam(0.001) )
    return model

# for one single Create a DNN model with only 'Horsepower' as input and horsepower_normalizer (Defined earlier) as the normalization layer

dnn_horsepower_model = build_and_compile_model(horsepower_normalizer)

dnn_horsepower_model.summary()

history = dnn_horsepower_model.fit(
    train_features['Horsepower'],
    train_labels,
    validation_split = 0.2,
    verbose=0, epochs=100
)

plot_loss(history)
plt.show()


x = tf.linspace(0.0, 250, 251)
y = dnn_horsepower_model.predict(x)

plot_horsepower(x, y)

# collecting the results on the test set for later
test_results['dnn_horsepower_model'] = dnn_horsepower_model.evaluate(
    test_features['Horsepower'], test_labels,
    verbose=0
)


# Selecting features excluding 'MPG' and 'Horsepower'
features = ['Cylinders', 'Displacement', 'Weight', 'Acceleration', 'Model Year', 'USA', 'Europe', 'Japan']

# Dictionary to store models and histories
regression_models = {}
histories = {}

# Iterate over each feature to build and train regression models
for feature in features:
    # Extract the feature values
    feature_values = np.array(train_features[feature])
    
    # Normalize the feature
    feature_normalizer = layers.Normalization(input_shape=[1,], axis=None)
    feature_normalizer.adapt(feature_values)
    
    # Build model
    model = tf.keras.Sequential([
        feature_normalizer,
        layers.Dense(units=1)
    ])
    
    # Compile model
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.1), loss='mean_absolute_error')
    
    # Train model
    history = model.fit(
        train_features[feature],
        train_labels,
        epochs=100,
        verbose=0,
        validation_split=0.2
    )
    
    # Store model and history
    regression_models[feature] = model
    histories[feature] = history

# Plotting loss for each feature model
plt.figure(figsize=(15, 10))
for feature, history in histories.items():
    plt.plot(history.history['loss'], label=f"{feature} loss")

plt.xlabel('Epoch')
plt.ylabel('Error [MPG]')
plt.title('Training Loss for Each Feature')
plt.legend()
plt.grid(True)
plt.show()


# Plotting predictions for each feature model
plt.figure(figsize=(15, 10))
for feature, model in regression_models.items():
    feature_values = train_features[feature].astype(np.float32)  # Convert to float32
    x = tf.linspace(tf.reduce_min(feature_values), tf.reduce_max(feature_values), 251)
    y = model.predict(x)
    plt.scatter(test_features[feature], test_labels, label=f"{feature} Data")
    plt.plot(x, y, label=f"{feature} Predictions")
    
plt.xlabel('Feature Value')
plt.ylabel('MPG')
plt.title('Predictions for Each Feature')
plt.legend()
plt.grid(True)
plt.show()
