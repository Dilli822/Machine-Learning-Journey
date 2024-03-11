
# building another model for cloth fashion classification
# TensorFlow and tf.keras
import tensorflow as tf
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


# loading the dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# class names of the cloth
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# exloreing the data
print("training image shapes --> n number of image, n x n pixel sizes ", train_images.shape)
print("Training labels ---> length of the labels ", len(train_labels))
print("Labels ---> ", train_labels)
print("------------ FOR TESTING ------")
print("Test Images Shape ---> ", test_images.shape)
print("Test Images Label ---> ", len(test_labels))



# Preprocess the data for image1 or first image from the tons of images
plt.figure()
plt.imshow(train_images[100])
plt.colorbar()
plt.grid(False)
plt.show()

# Scale these values to a range of 0 to 1 before feeding them to the neural network model.
train_images = train_images / 255.0
test_images = test_images / 255.0
# print("scaled train image between 0 to 1 ---> ", train_images)
# print("scaled between 0 to 1 test images", test_images)

# before building the neural network first we must check the n number of datas from the training model 
# let's display the first 35 images from the training set and display the class name below each image.

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
    
plt.show()


# ----------- BUILDING THE MODEL ---------------
# TO BUILD WE NEED LAYERS OF THE MODEL
# magic happens here
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# layers are the building block of a neural network
# layers extract representations from the data fed into them
# most deep learning consists of chaining together simple layers  tf.keras.layers.Dense parameters that are learned during the training
# The first layer in this network, tf.keras.layers.Flatten, transforms the format of
# the images from a two-dimensional array (of 28 by 28 pixels) to a one-dimensional array (of 28 * 28 = 784 pixels).
# Think of this layer as unstacking rows of pixels in the image and lining them up. This layer has no parameters to learn; it only reformats the data.


# After the pixels are flattened, the network consists of a sequence of two tf.keras.layers.Dense
# layers. These are densely connected, or fully connected, neural layers. The first Dense
# layer has 128 nodes (or neurons). The second (and last) layer returns a logits array with
# length of 10. Each node contains a score that indicates the current image belongs to one of the 10 classes.


#--------------- COMPILE THE MODEL -------------
# Optimizer - this is how the model is updated based on the data it sees and its loss function 
# Loss Function - This measures how accurate the model is during training, we have to minize the function to
# steer the model in the right direction
# Metrics --> used to monitor the training and testing steos,the following examples uses accuract, the fraction 
# of the images that are correctly classified

# "logits" refer to the raw scores or outputs produced by a model before applying a normalization or activation function.
#---- magic happens here----
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# -------- TRAIN THE MODEL --------
# we feed the train_images and train_labels arrays
# model learns to associate images and labels
# test_images array make a predictions about a test set
# test_labels array to verify that the predictions match the labels from test_labels array
# fitting the model magic happens here
model.fit(train_images, train_labels, epochs=10)


# evaluate the trained data
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

accuracy_graph = test_acc
loss_graph = 1 - test_acc

print('\n Test accuracy:', test_acc)
print('\n Test Loss:', test_loss)

# Epoch 1/10
# 1875/1875 [==============================] - 2s 747us/step - loss: 0.4944 - accuracy: 0.8271
# Epoch 2/10
# 1875/1875 [==============================] - 1s 751us/step - loss: 0.3771 - accuracy: 0.8637
# Epoch 3/10
# 1875/1875 [==============================] - 1s 742us/step - loss: 0.3406 - accuracy: 0.8767
# Epoch 4/10
# 1875/1875 [==============================] - 1s 738us/step - loss: 0.3130 - accuracy: 0.8856
# Epoch 5/10
# 1875/1875 [==============================] - 1s 763us/step - loss: 0.2973 - accuracy: 0.8904
# Epoch 6/10
# 1875/1875 [==============================] - 1s 746us/step - loss: 0.2825 - accuracy: 0.8954
# Epoch 7/10
# 1875/1875 [==============================] - 1s 744us/step - loss: 0.2702 - accuracy: 0.9003
# Epoch 8/10
# 1875/1875 [==============================] - 1s 736us/step - loss: 0.2590 - accuracy: 0.9051
# Epoch 9/10
# 1875/1875 [==============================] - 1s 743us/step - loss: 0.2483 - accuracy: 0.9076
# Epoch 10/10
# 1875/1875 [==============================] - 1s 763us/step - loss: 0.2395 - accuracy: 0.9100
# 313/313 - 0s - loss: 0.3341 - accuracy: 0.8821 - 231ms/epoch - 739us/step

# Test accuracy: 0.882099986076355 , 88.20%
pieLabels = ["Accuracy Percentage ", "Inaccuracy Percentage"]

# startangle by default it is x-axis 0 degree but we can override it
# legend
piedata = [accuracy_graph, loss_graph]
# Define the explode values (optional, if you want to explode some segments)
explode = (0.1, 0)  # Explode the first slice by 10% of the radius

# Plot the pie chart
# "explode" or separate a slice from the rest of the pie chart.
# In this code snippet, the autopct='%1.1f%%' parameter tells Matplotlib to display the 
# percentage values with one decimal place (%1.1f) followed by a percent sign (%%). Adjust the format string ('%1.1f%%') as needed to control the appearance of the percentage values.
plt.pie(piedata, labels=pieLabels, explode=explode, autopct='%1.1f%%', startangle=90)
plt.legend()  # Add legend
plt.show()
