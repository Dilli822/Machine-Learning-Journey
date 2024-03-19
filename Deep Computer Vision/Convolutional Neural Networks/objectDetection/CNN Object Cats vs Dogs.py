# Using a pretrained model
# https://chat.openai.com/share/387640c6-f467-4f5c-acce-b3d43ad6260c
# Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential

# Import the dataset
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# Define the URL for the dataset
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'

# Load and split the data into training, validation, and testing sets
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

# Define a function to get label names
get_label_name = metadata.features['label'].int2str

# Display a couple of images from the training dataset
for image, label in raw_train.take(2):
    plt.figure()
    plt.imshow(image)
    plt.title("Original Dataset: " + get_label_name(label))

plt.show()

# Resize the images to a consistent size (160x160) and preprocess them
IMG_SIZE = 160

def format_example(image, label):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label 

# Apply the formatting function to all images using map
train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

# Display the shape of resized images
for img, label in train.take(2):
    print("New shape: ", img.shape)

# Load the MobileNetV2 model as the base model
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)  # 3 for RGB channels
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE,
    include_top=False,  # Exclude the classification layer
    weights='imagenet'  # Use pre-trained weights from ImageNet
)

# Freeze the base model to prevent retraining of its weights
base_model.trainable = False

# Add a Global Average Pooling layer to reduce dimensionality
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

# Add a Dense layer for prediction (binary classification)
prediction_layer = keras.layers.Dense(1)

# Create the Sequential model by stacking layers together
model = Sequential([
    base_model,  # Pre-trained base model
    global_average_layer,  # Global Average Pooling layer
    prediction_layer  # Dense layer for prediction
])

# Display the summary of the model
model.summary()

