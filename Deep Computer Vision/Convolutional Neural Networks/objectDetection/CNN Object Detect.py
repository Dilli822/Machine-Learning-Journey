
"""
Coding Tutorials Link:
https://www.tensorflow.org/tutorials/images/cnn

1. Create our first Convnet to get familiar with CNN architectures.
2. Dataset - CIFAR Image Dataset wll be used in tensorflow to classify 10
   different everyday objects. total images 60K with 32 x 32 images
   with 6K images of each class.
   
   Labels in the dataset are:
   ['airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck']
    
    Dataset:
    Downloading data from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
   
   
"""


import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


# LOAD AND SPLIT THE DATASET
# loading tensorflow strange set of data objects
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize the pizel to between 0 and 1
# divide by 255 to make values between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names =    ['airplane', 'automobile', 'bird', 'cat', 'deer',
                  'dog', 'frog', 'horse', 'ship', 'truck']

# We can look at a image
IMG_INDEX = 7 # CHANGE NUMBERS TO SEE THE IMAGE

plt.imshow(train_images[IMG_INDEX], cmap=plt.cm.binary)
plt.xlabel(class_names[train_labels [IMG_INDEX] [0]])
plt.show()