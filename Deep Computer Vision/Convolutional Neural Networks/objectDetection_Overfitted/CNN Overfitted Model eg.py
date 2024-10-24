
# # Using a pretrained model
# # Link tutorial https://www.tensorflow.org/tutorials/images/transfer_learning

""" MOBILENET V2 """
import os
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import tensorflow as tf
keras = tf.keras 
from tensorflow.keras.models import Sequential
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# Datasets - load cats vs dogs dataset from module
# tensorflow dataset contains image and label pairs where
# images have different dimensions and 3 color channels
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
# split the data manually into 80% training, 10% testing
# and 10% validation
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

get_label_name = metadata.features['label'].int2str
# creates a function object that we can use to get labels

# display 2 images from the dataset 
for image, label in raw_train.take(1):
    plt.figure()
    plt.imshow(image)
    plt.title("Original Dataset " + get_label_name(label))

plt.show()

# lets reshape the images
# smaller the image better it is in compressed form
IMG_SIZE = 160 # All Images will be resized to 160 x 160

def format_example(image, label):
    """
    returns an image that is reshape to IMG_SIZE
    """
    image = tf.cast(image, tf.float32)
    image = (image/ 127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label 

# Now apply this function to all our image using map
train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

# lets have a look at our images now
for image, label in train.take(1):
    plt.figure()
    plt.imshow(image)
    plt.title("Training Dataset " + get_label_name(label))
    
plt.show()

# Now let's have a look at the shape of an original image vs the 
# new image we will see if it has been changed

for img, label in raw_train.take(2):
    print("Original Shape: ", img.shape)

for img, label in train.take(2):
    print("New shape: ", img.shape)
    
# here 3 is color channel
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(
    input_shape = IMG_SHAPE, # 160 x 160
    include_top = False, # do we include classifier ?
    weights = 'imagenet' # imagenet specific shape of weight
)

base_model.summary()

# setting base model False
base_model.trainable = False
base_model.summary()

# global average -  extracting meaningful features from the pre-trained convolutional base and preparing them for the final classification layers.
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

# Finally we will add the prediction layer that will be 
# a single dense neuron. We can do this because we only
# have two classes to predict for

# prediction layer 
prediction_layer = keras.layers.Dense(1)

# model layer
# -------- Magic Happens Here --------
from tensorflow.keras.models import Sequential

# Create the Sequential model
model = Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])

model.summary()

# Training the model - magic/compile happens here
# learning rate setting low means how much I am allowed to modify the weights 
# and biases of this network, which is what we have done setting low means
# we donot have to do major changes because we are already using a base model
# exists, right. So we'll set the learning rate.
# loss function is Binary since we are using two classes, if we were using more
# than two classes we would just have cross entropy or some other type of cross
# entropy. 
# RMSprop helps in efficiently training deep neural networks by adapting the learning rates,
# Binary Cross-Entropy is a loss function used for binary classification tasks that measures the difference 
# between predicted probabilities and actual labels

base_learning_rate = 0.001  # low is slower but precise

model.compile(
    optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate), #Image Classification:
    loss = tf.keras.losses.BinaryCrossentropy(from_logits = True), #Email Spam Detection:
    metrics = ['accuracy']
    )

# evaluation of model
# we can evaluate the model right now to see how it does before training it our 
# new images
initial_epochs = 3
validation_steps = 30

# Prepare batches for validation
# Define batch size
batch_size = 32

# Prepare batches
train_batches = train.shuffle(1280).batch(batch_size).prefetch(1)
validation_batches = validation.batch(batch_size).prefetch(1)

# loss0, accuracy0 = model.evaluate(validation_batches, steps=validation_steps)
# --- Below output is before the model training we evaluated upto 50%
# to evaluate at first comment out the model fit 
# 20/20 [==============================] - 7s 283ms/step - loss: 0.6981 - accuracy: 0.4984

# Train the model
history = model.fit(
    train_batches,
    epochs=initial_epochs,
    validation_data=validation_batches,
    validation_steps=validation_steps
)
acc = history.history['accuracy']
# Expected accuracy percentage is 93% from the tutorial
print(acc)
print(type(acc))
# print("Accuracy of the model is: "+  str(acc))
# print("inaccuracy is: " + str(1 - acc))


# explanation of 92% means pretty good we did using all original base layer like 1000 base layers 
# that classified upto 1000 different images, so very general applied to just cats vs dogs by adding
# dense layer classifier on the top

model.save("dogs_vs_cats.h5")
new_model = tf.keras.models.load_model("dogs_vs_cats.h5")


import matplotlib.pyplot as plt

# Plot training and validation accuracy over epochs
# Extract training and validation accuracy values
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Plot histograms for training and validation accuracy
plt.hist(train_accuracy, label='Training Accuracy', alpha=0.5, color='blue', bins=10)
plt.hist(val_accuracy, label='Validation Accuracy', alpha=0.5, color='orange', bins=10)
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.title('Distribution of Training and Validation Accuracy')
plt.show()


# Display some example images from the testing dataset along with model predictions
num_images = 10

for img, label in test.take(num_images):
    plt.figure()
    plt.imshow((img + 1) / 2)  # Rescale the image from [-1, 1] to [0, 1]
    plt.title(f'Label: {get_label_name(label.numpy())}')
    plt.axis('off')
    
    # Make a prediction using the loaded model
    img_input = tf.expand_dims(img, axis=0)  # Add batch dimension
    prediction = new_model.predict(img_input)
    predicted_label = "Dog" if prediction > 0 else "Cat"
    plt.text(10, 30, f'Model Prediction: {predicted_label}', color='red', fontsize=12, fontweight='bold')
    
    plt.show()
