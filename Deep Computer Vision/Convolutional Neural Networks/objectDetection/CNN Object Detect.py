import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# LOAD AND SPLIT THE DATASET
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

# Magic Happens Here 
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32 , 3)))
model.add(layers.MaxPooling2D( (2, 2) ))
model.add(layers.Conv2D(64, (3, 3), activation = 'relu' ))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3,3), activation = 'relu'))

model.summary()

model.add(layers.Flatten())
# 64 dense layers connected to activation function relu
model.add(layers.Dense(64, activation='relu'))
# output of 10 neural since we have 10 class of objects
model.add(layers.Dense(10))

model.summary()