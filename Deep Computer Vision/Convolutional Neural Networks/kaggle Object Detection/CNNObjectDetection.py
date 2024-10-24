import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
import numpy as np
import cv2
import requests
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

# Function to download and preprocess an image from a URL
def load_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.convert('RGB')  # Convert to RGB if it's not
    original_size = img.size  # Keep the original size for bounding box adjustment
    img = img.resize((224, 224))  # Resize the image to the model's expected input size
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array, original_size

# Function to draw bounding boxes on an image
def draw_bounding_box(image, bbox, original_size, color=(255, 0, 0), thickness=2):
    # Resize the bounding box according to the original image size
    orig_w, orig_h = original_size
    x_min, y_min, x_max, y_max = bbox
    x_min, y_min, x_max, y_max = int(x_min * orig_w), int(y_min * orig_h), int(x_max * orig_w), int(y_max * orig_h)
    
    return cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

# Function to display the image with the bounding box
def display_image_with_bbox(image_array, pred_bbox, original_size):
    img = image_array[0].astype('uint8')  # Remove batch dimension and convert to uint8
    img = cv2.resize(img, original_size)  # Resize the image back to original size
    img_with_bbox = draw_bounding_box(img, pred_bbox, original_size)

    # Display the image with the predicted bounding box
    plt.imshow(cv2.cvtColor(img_with_bbox, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Function to continuously allow URL input for object detection with error handling
def detect_objects_from_url(model):
    while True:
        try:
            # Input URL from user
            url = input("Enter an image URL (or type 'exit' to stop): ")
            if url.lower() == 'exit':
                break

            # Load the image from the URL
            img_array, original_size = load_image_from_url(url)

            # Make predictions
            pred_bbox = model.predict(img_array)[0]  # Predict bounding box

            # Display the image with the predicted bounding box
            display_image_with_bbox(img_array, pred_bbox, original_size)

        except Exception as e:
            # Print the error and reset the loop if any error occurs
            print(f"Error occurred: {e}. Please try again with a valid URL.")
            continue

# Function to build the object detection model
def build_model():
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze the base model layers

    # Add custom layers for bounding box prediction
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(4, activation='sigmoid')  # Output: 4 values for bounding box
    ])

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['mae'])
    return model

# Function to plot the training and validation loss
def plot_training_history(history):
    # Plot loss
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot accuracy (in this case, mean absolute error)
    plt.plot(history.history['mae'], label='train_mae')
    plt.plot(history.history['val_mae'], label='val_mae')
    plt.title('Training and Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.show()

# Load Pascal VOC 2007 dataset from TensorFlow Datasets
dataset, info = tfds.load('voc/2007', with_info=True, split='train', shuffle_files=True)

# Preprocess the dataset: resize images and normalize bounding boxes
def preprocess(data):
    image = data['image']
    bbox = data['objects']['bbox'][0]  # Use the first object's bbox
    
    # Resize the image to 224x224
    image = tf.image.resize(image, (224, 224))
    
    # Return the image and the bounding box
    return image, bbox

# Apply preprocessing to the dataset
dataset = dataset.map(preprocess)

# Batch the dataset and prefetch for performance
dataset = dataset.batch(32).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Split the dataset for training and validation
train_size = int(0.8 * len(dataset))
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

# Build the model
model = build_model()
model.summary()

# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10  # Adjust as needed
)

# Plot the training history (loss and accuracy)
plot_training_history(history)

# Start the loop to accept URLs and detect objects with error handling
detect_objects_from_url(model)


# https://upload.wikimedia.org/wikipedia/commons/1/11/Freightliner_M2_106_6x4_2014_%2814240376744%29.jpg
# https://upload.wikimedia.org/wikipedia/commons/d/de/Nokota_Horses_cropped.jpg
# https://anilblonnepal.wordpress.com/wp-content/uploads/2015/07/national-bird-of-nepal-danphe.jpg
# https://upload.wikimedia.org/wikipedia/commons/f/fc/Tarom.b737-700.yr-bgg.arp.jpg
# https://i.natgeofe.com/n/4f5aaece-3300-41a4-b2a8-ed2708a0a27c/domestic-dog_thumb_square.jpg
# https://upload.wikimedia.org/wikipedia/commons/3/39/European_Common_Frog_Rana_temporaria.jpg
# https://i.natgeofe.com/n/548467d8-c5f1-4551-9f58-6817a8d2c45e/NationalGeographic_2572187_square.jpg