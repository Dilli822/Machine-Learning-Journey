
# Using a pretrained model
# Link tutorial https://www.tensorflow.org/tutorials/images/transfer_learning
import os
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import tensorflow as tf
keras = tf.keras 

# Datasets - load cats vs dogs dataset from module
# tensorflow dataset contains image and label pairs where
# images have different dimensions and 3 color channels
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

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
for image, label in raw_train.take(4):
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
for image, label in train.take(2):
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

# output
# Original Shape:  (262, 350, 3)
# Original Shape:  (409, 336, 3)
# New shape:  (100, 100, 3)
# New shape:  (100, 100, 3)


"""
Picking a PreTrained Model
The model we are going to use as the convolutional base for
our model is the >> MobileNet V2 developed at Google. 
This model is trained on 1.4 million images and has
1000 different classes.

We want to use this model but only its convolutional base.
So wehn we load in the model we'll specify that we donot
want to load the top (classification) layer. We'll tell 
the model what input shape to extract and to use the 
predetermined weights from imagenet(Google dataset)

"""
# here 3 is color channel
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(
    input_shape = IMG_SHAPE, # 160 x 160
    include_top = False, # do we include classifier ?
    weights = 'imagenet' # imagenet specific shape of weight
)

base_model.summary()

# We are using this because we could not make it alone
# Ouput All this NN are the result of working by Experts,Ph.d SCholar

"""

output is : 
out_relu (ReLU)  (None, 5, 5, 1280)  0  ['Conv_1_bn[0][0]']
This o/p is actual useful and we gonna take this and pass
that to some more convolutional layers, and actually our 
classifier and use that to predict dogs versus cats.

At this point this base_mode will simply output a shape
(32. 5, 5, 1280) tensor that is a feature extraction
from our original (1, 160, 160, 3) image. This 32 means 
that we have 32 layers of different filters/features.

"""


# for image, _i in train_batches.take(1):
#     pass

# feature_batch = base_model(image)
# print(feature_batch.shape)
# # expected output is (32, 5,5,1280)
