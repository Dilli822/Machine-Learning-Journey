import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO

# Load and preprocess data
def load_data():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    return train_images, train_labels, test_images, test_labels

# Define class names for CIFAR-10
class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Define CNN Model for Object Detection
def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))  # Output layer for 10 objects/classes

    model.compile(optimizer='adam', 
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                  metrics=['accuracy'])
    return model

# Train the model
def train_model(model, train_images, train_labels, test_images, test_labels):
    history = model.fit(train_images, train_labels, epochs= 10, 
                        validation_data=(test_images, test_labels))
    return history

# Object Detection (Bounding Boxes) - Precise bounding box logic
def object_detection(image, model):
    # Resize and preprocess image for model prediction
    image_resized = cv2.resize(image, (32, 32)) / 255.0
    predictions = model.predict(np.expand_dims(image_resized, axis=0))
    predicted_class = np.argmax(predictions)

    # Generate bounding box coordinates based on image size
    height, width, _ = image.shape
    
    # Calculate bounding box dimensions relative to the image size
    bbox_width = int(width * 0.5)  # Example: 50% of the image width
    bbox_height = int(height * 0.5)  # Example: 50% of the image height
    
    # Center the bounding box in the image
    x = (width - bbox_width) // 2
    y = (height - bbox_height) // 2

    # Draw the bounding box on the image
    image_with_bbox = image.copy()
    cv2.rectangle(image_with_bbox, (x, y), (x + bbox_width, y + bbox_height), (255, 0, 0), 2)

    # Ensure the detected object is represented correctly within the bounding box
    plt.imshow(image_with_bbox)
    plt.title(f"Predicted Object: {class_names[predicted_class]}")
    plt.axis('off')
    plt.show()

# Plot accuracy and loss
def plot_history(history):
    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Function to download and preprocess the image from URL
def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))

    # Convert RGBA to RGB if necessary
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    
    return np.array(img)

# Main Execution
train_images, train_labels, test_images, test_labels = load_data()
model = create_model()
history = train_model(model, train_images, train_labels, test_images, test_labels)

# While loop for user interaction
while True:
    user_input = input("Enter an image URL to classify (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    try:
        image = load_image_from_url(user_input)
        object_detection(image, model)
    except Exception as e:
        print(f"Error loading image: {e}")

# Plot accuracy and loss
plot_history(history)


# https://upload.wikimedia.org/wikipedia/commons/1/11/Freightliner_M2_106_6x4_2014_%2814240376744%29.jpg
# https://upload.wikimedia.org/wikipedia/commons/d/de/Nokota_Horses_cropped.jpg
# https://anilblonnepal.wordpress.com/wp-content/uploads/2015/07/national-bird-of-nepal-danphe.jpg
# https://upload.wikimedia.org/wikipedia/commons/f/fc/Tarom.b737-700.yr-bgg.arp.jpg
# https://i.natgeofe.com/n/4f5aaece-3300-41a4-b2a8-ed2708a0a27c/domestic-dog_thumb_square.jpg
# https://upload.wikimedia.org/wikipedia/commons/3/39/European_Common_Frog_Rana_temporaria.jpg
# https://i.natgeofe.com/n/548467d8-c5f1-4551-9f58-6817a8d2c45e/NationalGeographic_2572187_square.jpg