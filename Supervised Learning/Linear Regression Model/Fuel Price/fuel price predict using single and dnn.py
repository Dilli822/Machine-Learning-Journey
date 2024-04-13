import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load dataset
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(url, names=column_names, na_values='?', comment='\t', sep=' ', skipinitialspace=True)
dataset = raw_dataset.copy().dropna()

# Data preprocessing
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Split features from labels
train_features = train_dataset.copy()
test_features = test_dataset.copy()
train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

# Normalization
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features, dtype=np.float32))

# Linear Regression with One Variable (Horsepower)
horsepower = np.array(train_features['Horsepower'])
horsepower_normalizer = layers.Normalization(input_shape=[1,], axis=None)
horsepower_normalizer.adapt(horsepower)

horsepower_model = tf.keras.Sequential([
    horsepower_normalizer,
    layers.Dense(units=1)
])

horsepower_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.1), loss="mean_absolute_error")

history = horsepower_model.fit(
    train_features['Horsepower'],
    train_labels,
    epochs=100,
    verbose=0,
    validation_split=0.2
)

# Regression with DNN
def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.legacy.Adam(0.001))
    return model

dnn_horsepower_model = build_and_compile_model(horsepower_normalizer)
history = dnn_horsepower_model.fit(
    train_features['Horsepower'],
    train_labels,
    validation_split=0.2,
    verbose=0,
    epochs=100
)

# Regression with other features
features = ['Cylinders', 'Displacement', 'Weight', 'Acceleration', 'Model Year', 'USA', 'Europe', 'Japan']
regression_models = {}

for feature in features:
    feature_values = np.array(train_features[feature])
    feature_normalizer = layers.Normalization(input_shape=[1,], axis=None)
    feature_normalizer.adapt(feature_values)
    
    model = tf.keras.Sequential([
        feature_normalizer,
        layers.Dense(units=1)
    ])
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.1), loss='mean_absolute_error')
    
    history = model.fit(
        train_features[feature],
        train_labels,
        epochs=100,
        verbose=0,
        validation_split=0.2
    )
    regression_models[feature] = model
