
# url = "https://www.shutterstock.com/shutterstock/videos/23708860/thumb/1.jpg"
# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image
import requests
from PIL import Image
from io import BytesIO

# Load the dataset (using MNIST for handwritten digit recognition)
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Reshape the data to add a single channel (grayscale)
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# Build the CNN model
model = models.Sequential()

# Add layers: Convolution + Pooling
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Add Fully Connected layers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))  # 10 classes for digits 0-9

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model with validation split to detect overfitting or underfitting
history = model.fit(train_images, train_labels, epochs=15, validation_split=0.2)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"Test accuracy: {test_acc}")

# Get the final training and validation accuracy from history
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]

# Check for overfitting, underfitting, or normal fit
def check_model_fit(train_acc, val_acc):
    print(f"\nFinal Training Accuracy: {train_acc:.4f}")
    print(f"Final Validation Accuracy: {val_acc:.4f}")

    if train_acc > val_acc + 0.1:
        print("\nThe model is overfitting.")
    elif train_acc < 0.7 and val_acc < 0.7:
        print("\nThe model is underfitting.")
    else:
        print("\nThe model has a normal fit.")

# Call the function to check model fit
check_model_fit(train_acc, val_acc)

# Plot accuracy and loss
def plot_accuracy_loss(history):
    epochs = range(1, len(history.history['accuracy']) + 1)

    # Plot accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history['accuracy'], label='Training Accuracy')
    plt.plot(epochs, history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Plot loss (Gradient Descent)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history['loss'], label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()

plot_accuracy_loss(history)

# Function to predict a custom image from URL
def predict_custom_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('L')  # Convert image to grayscale
    img = img.resize((28, 28))  # Resize to 28x28
    img = np.array(img) / 255.0  # Normalize the image
    img = img.reshape(1, 28, 28, 1)  # Reshape to add batch dimension and channel

    # Make prediction
    prediction = model.predict(img)
    predicted_label = np.argmax(prediction)
    
    plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)
    plt.title(f"Predicted Digit: {predicted_label}")
    plt.show()

# Allow user to input the URL for custom digit image
# url = input("Enter the URL of a handwritten digit image: ")
# predict_custom_image(url)

# https://1001freedownloads.s3.amazonaws.com/vector/thumb/64381/0-9-Handwritten-5.png
# https://i.sstatic.net/CF1ze.jpg
# https://thumb.ac-illust.com/51/513ad2b0cfad8ad3dce41e82ad5150c4_t.jpeg
# https://thumb.ac-illust.com/c9/c9b9da8c6dce5f66ea6a17ab3a46b91e_t.jpeg
# Terminal options loop
while True:
    print("\nOptions:")
    print("1: Enter the URL of a handwritten digit image")
    print("2: Exit")
    choice = input("Choose an option (1 or 2): ")

    if choice == '1':
        url = input("Enter the URL of a handwritten digit image: ")
        predict_custom_image(url)
    elif choice == '2':
        print("Exiting the program.")
        break
    else:
        print("Invalid choice. Please enter 1 or 2.")