import pytesseract
from PIL import Image
import cv2
import numpy as np
import os
import requests
from io import BytesIO

def check_tesseract():
    try:
        pytesseract.get_tesseract_version()
        print("Tesseract is correctly installed.")
    except pytesseract.TesseractNotFoundError:
        print("Tesseract is not installed or not found. Please check your installation.")
        return False
    return True

def get_image(image_path_or_url):
    if image_path_or_url.startswith(('http://', 'https://')):
        # It's a URL
        response = requests.get(image_path_or_url)
        img = Image.open(BytesIO(response.content))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    else:
        # It's a local file path
        if not os.path.exists(image_path_or_url):
            print(f"Image file not found: {image_path_or_url}")
            return None
        return cv2.imread(image_path_or_url)

def preprocess_image(img):
    if img is None:
        print("Failed to read image.")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    return gray

def extract_text_from_image(image_path_or_url):
    if not check_tesseract():
        return None

    img = get_image(image_path_or_url)
    if img is None:
        return None

    processed_image = preprocess_image(img)
    if processed_image is None:
        return None

    try:
        text = pytesseract.image_to_string(processed_image)
        if not text.strip():
            print("No text was extracted. The image might be unclear or contain no readable text.")
        return text
    except Exception as e:
        print(f"An error occurred during text extraction: {str(e)}")
        return None

# # Example usage
image_source = 'captured_image.jpg'  # Can be a URL or local file path
extracted_text = extract_text_from_image(image_source)
if extracted_text:
    print("Extracted text:")
    print(extracted_text)
else:
    print("Failed to extract text.")

# import cv2
# import pytesseract

# # Specify the path to the Tesseract-OCR executable if it's not in your PATH
# # Uncomment and update the following line based on your installation
# # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows
# # pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'  # Mac

# def capture_image_on_spacebar():
#     # Initialize the camera
#     cap = cv2.VideoCapture(0)

#     if not cap.isOpened():
#         print("Error: Could not open camera.")
#         return

#     print("Press the spacebar to capture an image, and 'q' to quit.")

#     while True:
#         # Capture frame-by-frame
#         ret, frame = cap.read()

#         if not ret:
#             print("Error: Could not capture image.")
#             break

#         # Display the resulting frame
#         cv2.imshow("Camera", frame)

#         # Wait for user input
#         key = cv2.waitKey(1)

#         # Check if the spacebar is pressed
#         if key == 32:  # Spacebar key
#             # Save the captured image
#             cv2.imwrite('captured_image.jpg', frame)
#             print("Image captured and saved as 'captured_image.jpg'.")

#             # Release the camera and close the window after capturing
#             cap.release()
#             cv2.destroyAllWindows()
#             extract_text_from_image('captured_image.jpg')  # Extract text from the captured image
#             break  # Exit the loop

#         # Check if 'q' key is pressed to quit
#         elif key == ord('q'):
#             print("Exiting the camera.")
#             cap.release()  # Release the camera
#             cv2.destroyAllWindows()  # Close the window
#             break  # Exit the loop

# def extract_text_from_image(image_path):
#     # Read the image using OpenCV
#     img = cv2.imread(image_path)

#     # Convert the image to RGB (Tesseract expects RGB images)
#     rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     # Use pytesseract to do OCR on the image
#     text = pytesseract.image_to_string(rgb_image)

#     # Print the extracted text
#     print("Extracted Text:")
#     print(text)

# if __name__ == "__main__":
#     capture_image_on_spacebar()


# import cv2
# import pytesseract

# # Specify the path to the Tesseract-OCR executable if it's not in your PATH
# # Uncomment and update the following line based on your installation
# # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows
# # pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'  # Mac

# def capture_image_on_spacebar():
#     """Capture an image when the spacebar is pressed."""
#     # Initialize the camera
#     cap = cv2.VideoCapture(0)

#     if not cap.isOpened():
#         print("Error: Could not open camera.")
#         return

#     print("Press the spacebar to capture an image, and 'q' to quit.")

#     while True:
#         # Capture frame-by-frame
#         ret, frame = cap.read()

#         if not ret:
#             print("Error: Could not capture image.")
#             break

#         # Display the resulting frame
#         cv2.imshow("Camera", frame)

#         # Wait for user input
#         key = cv2.waitKey(1)

#         # Check if the spacebar is pressed
#         if key == 32:  # Spacebar key
#             # Save the captured image
#             cv2.imwrite('captured_image.jpg', frame)
#             print("Image captured and saved as 'captured_image.jpg'.")

#             # Release the camera and close the window after capturing
#             cap.release()
#             cv2.destroyAllWindows()
#             extract_text_from_image('captured_image.jpg')  # Extract text from the captured image
#             break  # Exit the loop

#         # Check if 'q' key is pressed to quit
#         elif key == ord('q'):
#             print("Exiting the camera.")
#             cap.release()  # Release the camera
#             cv2.destroyAllWindows()  # Close the window
#             break  # Exit the loop

# def extract_text_from_image(image_path):
#     """Extract text from the given image file."""
#     # Read the image using OpenCV
#     img = cv2.imread(image_path)

#     # Convert the image to RGB (Tesseract expects RGB images)
#     rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     # Use pytesseract to get data on the image
#     data = pytesseract.image_to_data(rgb_image, output_type=pytesseract.Output.DICT)

#     # Extract and print each word with its position and confidence score
#     print("Extracted Text and Details:")
#     n_boxes = len(data['text'])
#     for i in range(n_boxes):
#         if int(data['conf'][i]) > 0:  # Only consider positive confidence
#             word = data['text'][i]
#             if word.strip():  # Check if the word is not empty
#                 x = data['left'][i]
#                 y = data['top'][i]
#                 w = data['width'][i]
#                 h = data['height'][i]
#                 print(f"Word: '{word}', Position: ({x}, {y}), Size: ({w}x{h}), Confidence: {data['conf'][i]}")

# if __name__ == "__main__":
#     capture_image_on_spacebar()



# import cv2
# import pytesseract

# # Specify the path to the Tesseract-OCR executable if it's not in your PATH
# # Uncomment and update the following line based on your installation
# # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows
# # pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'  # Mac

# def capture_image_on_spacebar():
#     """Capture an image when the spacebar is pressed."""
#     # Initialize the camera
#     cap = cv2.VideoCapture(0)

#     if not cap.isOpened():
#         print("Error: Could not open camera.")
#         return

#     print("Press the spacebar to capture an image, and 'q' to quit.")

#     while True:
#         # Capture frame-by-frame
#         ret, frame = cap.read()

#         if not ret:
#             print("Error: Could not capture image.")
#             break

#         # Display the resulting frame
#         cv2.imshow("Camera", frame)

#         # Wait for user input
#         key = cv2.waitKey(1)

#         # Check if the spacebar is pressed
#         if key == 32:  # Spacebar key
#             # Save the captured image
#             cv2.imwrite('captured_image.jpg', frame)
#             print("Image captured and saved as 'captured_image.jpg'.")

#             # Release the camera and close the window after capturing
#             cap.release()
#             cv2.destroyAllWindows()
#             extract_text_from_image('captured_image.jpg')  # Extract text from the captured image
#             break  # Exit the loop

#         # Check if 'q' key is pressed to quit
#         elif key == ord('q'):
#             print("Exiting the camera.")
#             cap.release()  # Release the camera
#             cv2.destroyAllWindows()  # Close the window
#             break  # Exit the loop

# def extract_text_from_image(image_path):
#     """Extract text from the given image file."""
#     # Read the image using OpenCV
#     img = cv2.imread(image_path)

#     # Convert the image to RGB (Tesseract expects RGB images)
#     rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     # Use pytesseract to get data on the image
#     data = pytesseract.image_to_data(rgb_image, output_type=pytesseract.Output.DICT)

#     # Prepare a list to hold extracted text
#     extracted_words = []

#     # Extract and save each word
#     n_boxes = len(data['text'])
#     for i in range(n_boxes):
#         if int(data['conf'][i]) > 0:  # Only consider positive confidence
#             word = data['text'][i]
#             if word.strip():  # Check if the word is not empty
#                 extracted_words.append(word)

#     # Combine the extracted words into a single string
#     extracted_text = ' '.join(extracted_words)

#     # Print and save the extracted text
#     print("Extracted Text:")
#     print(extracted_text)

#     # Save the extracted text to a file
#     with open('extracted_text.txt', 'w') as text_file:
#         text_file.write(extracted_text)
#         print("Extracted text saved as 'extracted_text.txt'.")

# if __name__ == "__main__":
#     capture_image_on_spacebar()



# import cv2
# import pytesseract
# import pyttsx3

# # Specify the path to the Tesseract-OCR executable if it's not in your PATH
# # Uncomment and update the following line based on your installation
# # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows
# # pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'  # Mac

# def capture_image_on_spacebar():
#     """Capture an image when the spacebar is pressed."""
#     # Initialize the camera
#     cap = cv2.VideoCapture(0)

#     if not cap.isOpened():
#         print("Error: Could not open camera.")
#         return

#     print("Press the spacebar to capture an image, and 'q' to quit.")

#     while True:
#         # Capture frame-by-frame
#         ret, frame = cap.read()

#         if not ret:
#             print("Error: Could not capture image.")
#             break

#         # Display the resulting frame
#         cv2.imshow("Camera", frame)

#         # Wait for user input
#         key = cv2.waitKey(1)

#         # Check if the spacebar is pressed
#         if key == 32:  # Spacebar key
#             # Save the captured image
#             cv2.imwrite('captured_image.jpg', frame)
#             print("Image captured and saved as 'captured_image.jpg'.")

#             # Release the camera and close the window after capturing
#             cap.release()
#             cv2.destroyAllWindows()
#             extract_text_from_image('captured_image.jpg')  # Extract text from the captured image
#             break  # Exit the loop

#         # Check if 'q' key is pressed to quit
#         elif key == ord('q'):
#             print("Exiting the camera.")
#             cap.release()  # Release the camera
#             cv2.destroyAllWindows()  # Close the window
#             break  # Exit the loop

# def extract_text_from_image(image_path):
#     """Extract text from the given image file."""
#     # Read the image using OpenCV
#     img = cv2.imread(image_path)

#     # Convert the image to RGB (Tesseract expects RGB images)
#     rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     # Use pytesseract to get data on the image
#     data = pytesseract.image_to_data(rgb_image, output_type=pytesseract.Output.DICT)

#     # Prepare a list to hold extracted text
#     extracted_words = []

#     # Extract and save each word
#     n_boxes = len(data['text'])
#     for i in range(n_boxes):
#         if int(data['conf'][i]) > 0:  # Only consider positive confidence
#             word = data['text'][i]
#             if word.strip():  # Check if the word is not empty
#                 extracted_words.append(word)

#     # Combine the extracted words into a single string
#     extracted_text = ' '.join(extracted_words)

#     # Print and save the extracted text
#     print("Extracted Text:")
#     print(extracted_text)

#     # Save the extracted text to a file
#     with open('extracted_text.txt', 'w') as text_file:
#         text_file.write(extracted_text)
#         print("Extracted text saved as 'extracted_text.txt'.")

#     # Speak the extracted text
#     speak_text(extracted_text)

# def speak_text(text):
#     """Convert the given text to speech."""
#     engine = pyttsx3.init()  # Initialize the TTS engine
#     engine.setProperty('rate', 150)  # Set the speech rate (default is usually around 200)
#     engine.say(text)              # Queue the text to be spoken
#     engine.runAndWait()          # Wait until the speaking is finished

# if __name__ == "__main__":
#     capture_image_on_spacebar()
