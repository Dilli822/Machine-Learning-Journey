
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
# plt.show()

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

""" 
Training Model 

"""
# model compilation with optimizer 
# Magic Happens Here
# optimizer is adam
# loss function used here is SparseCategoricalCrossentropy as classification task
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=1,
                    validation_data = (test_images, test_labels))

# loss: 0.9012 - accuracy: 0.6836 - val_loss: 0.9445 - val_accuracy: 0.6711


# Evaluating the Model - to determine how well the model performed by 
# looking at it's performance on the test data set

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(test_acc)
# approax model accuracy we have 70% as the dataset is small only 60k


# so we use techniques to make our model accuracy better
"""
    ---- DATA AUGMENTATION ----
    To avoid overfitting and create a larger dataset from a smaller one
    we can use a techinque called data augmentation. This is simply
    performing random transformations on our images so that model can
    generalize better. These transformations can be things like
    compressions, relations, stretches and even color changes.
    
    To create a very good compositioal neural network from scartch, if 
    we are using a small dataset we employ data augmentation.
    
    Fortunately keras can help us do this. Look at the code below to 
    an example of data augmentation.
"""


"""
Data augmentation means just making our limited images stretch, flipped
rotate just to make our image in the differnt shapes or views without 
changing its essence that helps to generalize the original image 
or training data
"""

from keras.preprocessing import image 
from keras.preprocessing.image import ImageDataGenerator

# Create a data generator object that transforms images
datagen = ImageDataGenerator(
    rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'nearest'
)
 

# pick an image to transform
test_img = train_images[14]
img = image.img_to_array(test_img) # convert image to numpy array
img = img.reshape((1, ) + img.shape) # reshape image

i = 0

# loops runs forever untill we break, saving the images to current dict
# 4 times for each images makes 40,000 for each 10,000 original images
for batch in datagen.flow(img, save_prefix='text', save_format='jpeg'):
    plt.figure(i)
    plot = plt.imshow(image.img_to_array(batch[0]))
    i += 1
    if i < 4:
        break 
    
plt.show()

