
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
# plt.show()

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
    
# plt.show()


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

# ----------------- MAKE PREDICTIONS WITH THE MODEL ----------------
# Attaching a softmax layer to convert the model's linear output -logits to probablities
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

predictions[0]

print("Predictions array is of 10 numbers because of 10 classes in our model ")
print("Predictions for first class ")
print(predictions[0])
# 1.7628969e-12 1.4097951e-11 2.0145526e-11 1.9683134e-11 7.1194134e-11
#  1.6901217e-04 3.2140952e-09 1.7161971e-03 3.2127958e-09 9.9811482e-01]
print(predictions[5])
# [9.2641103e-06 9.9998891e-01 8.0150147e-07 1.9066354e-08 1.0075415e-06
#  3.1658253e-13 8.4052960e-09 3.8918243e-14 1.4398661e-11 2.2756234e-16]
# from above 5 index 9.26 is highest prediction value that means it must be T-shirt/top class cloth

# model should be confident here which class of cloth is this
np.argmax(predictions[2])
test_labels[2]

# functions to graph the full set of 20 class predictions
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    # used to set the locations of the tick marks on the respective axes.
    # it removes the tick marks altogether.
    plt.xticks([]) 
    plt.yticks([])
    
    plt.imshow(img, cmap=plt.cm.binary) # to set the colormap to binary
    
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = "green"
    else:
        color = "red"
        
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)
    plt.title("Image {}".format(i))  # Add this line to display the index of the image


def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(20))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('green')
    

# Verify predictions
# we have set green color for correct predictions and red color for incorrect

i = 0 # or first image
plt.figure(figsize=(6,3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
# plt.show()


i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
# plt.show()

# Verifying for multiple images at once
# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
# plt.show()


# LETS USE TRAINED MODEL
img = test_images[1]
print(img.shape)

img = (np.expand_dims(img, 0))
print(img.shape)

predictions_single = probability_model.predict(img)
print(predictions_single)

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()

np.argmax(predictions_single[0])


accuracy_graph = test_acc
loss_graph = 1 - test_acc
points = [accuracy_graph, loss_graph]
colors = ["r", "g"]
explodes = [0.5, 0]

pieLabels = ["Accuracy Percentage ", "Inaccuracy Percentage"]
print('\n Test accuracy:', test_acc)
print('\n Test Loss:', test_loss)

plt.pie(points,labels=pieLabels, explode=explodes, colors=colors, shadow=True,startangle=40, autopct='%1.1f%%' )
plt.legend(title="Model Percentage")
plt.show()