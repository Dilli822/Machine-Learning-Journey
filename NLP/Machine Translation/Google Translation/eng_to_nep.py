import cv2
import pytesseract
from googletrans import Translator
from gtts import gTTS
import os
import subprocess
import platform

# Load the image
image_path = 'scanned_image.png'  # Replace with your image file
img = cv2.imread(image_path)

# Check if the image was successfully loaded
if img is None:
    print(f"Error: Could not open or read the image {image_path}")
    exit()

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Use pytesseract to extract text from the image
extracted_text = pytesseract.image_to_string(gray)

# Display the extracted text
print("Extracted Text:")
print(extracted_text)

# Translate the extracted text to Nepali using googletrans
translator = Translator()
translated_text = translator.translate(extracted_text, dest='ne').text

# Display the translated text
print("Translated Text (Nepali):")
print(translated_text)

# Convert the translated text to speech using gTTS
tts = gTTS(text=translated_text, lang='ne')
tts.save('nepali_speech.mp3')

# Play the speech with loud volume
def play_audio(file):
    # Check the operating system
    if platform.system() == "Darwin":  # macOS
        subprocess.call(["afplay", file])
    elif platform.system() == "Linux":
        subprocess.call(["mpg123", "-f", "80000", file])  # Change to a loud volume
    elif platform.system() == "Windows":
        os.startfile(file)  # This may not have volume control, but works on Windows
    else:
        print("Unsupported OS")

play_audio('nepali_speech.mp3')
