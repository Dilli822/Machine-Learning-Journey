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


"""
--- CNN Architecture
A common architecture for a CNN is a stack of Conv2D layers followed by a few
densely conected layers. To idea is that the stack of convolutional and maxPooling
layers extract the features from the image. The these features are flattened
and fed to densly connected layers that determine the class of an image based 
on the presence of features.

We will start by building the Convolutional Base.

"""
# Magic Happens Here 
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32 , 3)))
model.add(layers.MaxPooling2D( (2, 2) ))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu' ))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3,3), activation = 'relu'))

"""
-- Layer 1
The input shape of our data will be 32 x 32 and 3 and we will process 32 filters
of size 3 x 3 over over input data. We will also apply the activation function
relut to the output of each convolution operation.

-- Layer 2
This layer will perform the max pooling operation using 2 x 2 samples and 
a stride of 2.

-- Other Layers
The next set of layers do very similar things but takes as input feature map
from the previous layer. They also increase the frequency of fikters from
32 to 64. We can do this as our data shrinks in special dimensions as it passed
through the layers, meaning we can afford (computationally) to add more depth.

"""

model.summary() # let's have a look at our model so far

# Output 
# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  conv2d (Conv2D)             (None, 30, 30, 32)        896       
                                                                 
#  max_pooling2d (MaxPooling2  (None, 15, 15, 32)        0         
#  D)                                                              
                                                                 
#  conv2d_1 (Conv2D)           (None, 13, 13, 64)        18496     
                                                                 
#  max_pooling2d_1 (MaxPoolin  (None, 6, 6, 64)          0         
#  g2D)                                                            
                                                                 
#  conv2d_2 (Conv2D)           (None, 4, 4, 64)          36928     
                                                                 
# =================================================================
# Total params: 56320 (220.00 KB)
# Trainable params: 56320 (220.00 KB)
# Non-trainable params: 0 (0.00 Byte)

"""
conv2d Output Shape is 30 x 30 because we are using 2 pixels as padding
 - 1. max_pooling2d (MaxPooling2D)    (None, 15, 15, 32)        0         
 - 2. 30 shrunk by factor 2 
 - 3. then again we do convt on 15 15 32 and we get
 - conv2d_1 (Conv2D)              (None, 13, 13, 64)        18496
 - 4.again divide step 3. by factor of 2 we get
 - max_pooling2d_1 (MaxPooling2D)    (None, 6, 6, 64)          0 
 - 5. conv2d_2 (Conv2D)           (None, 4, 4, 64)          36928 
 4 x 4 put all those stacks in straight line 1D
 
 These output only describes the presence of specific features as we've gone
 through convt base which is called the stack of convt and max pooling layers.
 
 Now what actually we need to do is pass this information into some kind 
 of dense layer classifier, which is actually going to take this pixel 
 data that we've kind of calculated and found, so the almost extraction of
 features that exist in the image,and tell us which combination of these
 features map to either you know, what one of these 10 classes are.
 So that's kind of the point we do this convt base, which extracts all of the
 features out of our image. And then we use thte dense network to say, Okay,well
 these combination of features exist, then that means this image is this,otherwise
 it's this and that, and so on. So that's we are doing here. Alright, so
 let's say adding the dense layer. So to add the dense layer is pretty easy
 model  .Flatten() 
"""

# Adding Dense Layers
# So far we have just completed the convolutional base. Now we need to take
# these extracted features and add a way to classify them. This is why we add
# the following layers to our model.

model.add(layers.Flatten())
# 64 dense layers connected to activation function relu
model.add(layers.Dense(64, activation='relu'))
# output of 10 neural since we have 10 class of objects
model.add(layers.Dense(10))

model.summary()

# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  conv2d (Conv2D)             (None, 30, 30, 32)        896       
                                                                 
#  max_pooling2d (MaxPooling2  (None, 15, 15, 32)        0         
#  D)                                                              
                                                                 
#  conv2d_1 (Conv2D)           (None, 13, 13, 64)        18496     
                                                                 
#  max_pooling2d_1 (MaxPoolin  (None, 6, 6, 64)          0         
#  g2D)                                                            
                                                                 
#  conv2d_2 (Conv2D)           (None, 4, 4, 64)          36928     
                                                                 
#  flatten (Flatten)           (None, 1024)              0         
                                                                 
#  dense (Dense)               (None, 64)                65600     
                                                                 
#  dense_1 (Dense)             (None, 10)                650       
                                                                 
# =================================================================
# Total params: 122570 (478.79 KB)
# Trainable params: 122570 (478.79 KB)
# Non-trainable params: 0 (0.00 Byte)


# Interpretation of output 
#  flatten (Flatten)           (None, 1024)              0     
# - flatten that is 1042 which 64 x 4 times                                                       
#  dense (Dense)               (None, 64)                65600   
# - and the dense layer is 64                                                         
#  dense_1 (Dense)             (None, 10)                650
# output dense layer 10   
    
# we can see that the flatten layer changes the shape of our data
# so that we can feed it to the 64 node dense layer, followed by the
# final output layer of 10 neurons (one for each class)        

"""
-- Pretrained Model 
-- We would have noticed that the model takes a few minutes to train in the
   Notebook and only giv'es an accuraccy of ~ 70% this is okay but surely
   there is a way to improve on this.
   In this section, we will talk about using a pretrained CNN as apart of
   our own custom network to improve the accuracy of our model. We know
   that CNN's alone (with no dense layers) donot do anything that map the
   presence of features from our input. This means we can use a pretrained 
   CNN, one trained on millions of images, as the start of our model. This
   allows us to have a very good classifier for a relatively smaller
   dataset ( < 10, 000 images). This is because the convnet already has
   a very good idea of what features to look for an image and can find 
   them very effectively. So if we can determine the presence of features
   that all the rest of the model needds to do is determine which combination
   of features makes a specific image.
   
   
   Explaination:
   We have used augmenation, that's great technique if we want to increase
   the size of our data set. But what if even after that we still donot
   have enough images in our dataset?
   well, what we have can do is use something called a pre trained model
   now companies like google, and tensorflow, which is owned by Google
   make their own amazing compositional neural networks that are completely
   open source that we can use. So what we are going to do is actually use
   part of a Convolutional neural network that they've trained already on,
   I guess 1.4 million images,And we are just going to use part of that 
   model, as kind of the base of our models so that we have a really good
   starting point. And all we need to do is called fine tune the last few
   layers of that networkm so that they work a little bit better for our
   purposes. So what we are going to do essentially say, all right 
   we have this model that Google's trained they've trained it on 1.4
   million images its capable of classifying, let's say 1000 different classes
   which is actually the example we'll look at later.
    
"""


"""
   Fine Tuning 
   
   When we employ the techique defined above we will often want to tweak the 
   final layers in our convolutional base to work better for our specific
   problem. This involves not touching or retraining the earlier layers in 
   our convolutional base but only adjusting the final few. We do this
   because the first layers in our base are very good at extracting low
   level features like lines and egdes, things that are similar for any 
   kind of image. 
   where the later layers are better at picking up very specific features
   like lines and edges, things that are similar for any kind of image.
   where the later layers are better at picking up very specific features
   like shapes or even eyes. 
   >>>> If we adjust the final layers that we can look for only features 
   relevant to our specific problem.
   
   
   Explanation Continues....... 
   
   So obviously the begining of that model, is what's picking up on the
   smaller edges, and you know kind of the very general things that appear
   in all of our images. So if we can use the base of that model, so kind 
   of the  begining of it, that does a really good job picking up on edges,
   and general things that will apply to any image, then what we can do
   is just change the top layers of that model a tiny bit, or add our own
   layers to it to classify for the problem that we want. And that should be
   a very effective wy to use this pre trained model, we are saying we are
   going to use the begining part that's really good at kind of the 
   generalization step, then we'll pass it into our own layets that will
   do whatever we need to do specifically for our problem. That's what's 
   like fine tuning step. And then we should have a model that works pretty 
   well. in fact that's what we are going to do in this example, now.
   
   So that's kind of the point of what I am talking about here is using 
   part of a model that already exists that's very good at generalizing
   it's been trained on so many different images. And then we'll pass 
   our own training data in, we won't modify the beginning aspect of 
   our neural network, because it already works really well, we'll
   just modify that last few layers that are really good at classifying,
   for example, just cats and dogs, which is exactly the example
   we are actually going to do here. 

"""

# Using a pretrained model
# Link tutorial https://www.tensorflow.org/tutorials/images/transfer_learning
import os
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import tensorflow as tf

_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

BATCH_SIZE = 32
IMG_SIZE = (160, 160)

train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                            shuffle=True,
                                                            batch_size=BATCH_SIZE,                                                            
                                                            image_size=IMG_SIZE)
          




                                                     