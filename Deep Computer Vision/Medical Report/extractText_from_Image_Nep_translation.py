import cv2
import pytesseract
import requests
import numpy as np
from googletrans import Translator
from gtts import gTTS
import os
from pydub import AudioSegment
from pydub.playback import play

# Specify the path to the Tesseract-OCR executable if it's not in your PATH
# Uncomment and update the following line based on your installation
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows
# pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'  # Mac

def download_image(url):
    """Download an image from a URL."""
    response = requests.get(url)
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return img

def preprocess_image(image):
    """Convert the image to grayscale and apply thresholding."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply binary thresholding
    _, thresh_img = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    return thresh_img

def extract_text_from_image(image):
    """Extract text from the given image."""
    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Use pytesseract to get data on the image
    data = pytesseract.image_to_data(preprocessed_image, output_type=pytesseract.Output.DICT)

    # Prepare a list to hold extracted text
    extracted_words = []

    # Extract and save each word
    n_boxes = len(data['text'])
    for i in range(n_boxes):
        if int(data['conf'][i]) > 0:  # Only consider positive confidence
            word = data['text'][i]
            if word.strip():  # Check if the word is not empty
                extracted_words.append(word)

    # Combine the extracted words into a single string
    extracted_text = ' '.join(extracted_words)

    # Print and save the extracted text
    print("Extracted Text:")
    print(extracted_text)

    # Save the extracted text to a file
    with open('extracted_text.txt', 'w') as text_file:
        text_file.write(extracted_text)
        print("Extracted text saved as 'extracted_text.txt'.")

    return extracted_text

def translate_text(text, dest_language='ne'):
    """Translate the given text to the specified language."""
    translator = Translator()
    translated = translator.translate(text, dest=dest_language)
    return translated.text

def speak_text(text, language='ne', volume_db=0):
    """Convert the given text to speech in the specified language and play it loudly."""
    tts = gTTS(text=text, lang=language, slow=False)
    audio_file = "translated_audio.mp3"
    tts.save(audio_file)  # Save the audio file

    # Load the audio file with pydub
    audio_segment = AudioSegment.from_file(audio_file)

    # Increase the volume
    loud_audio = audio_segment + volume_db  # Increase the volume (in dB)

    # Play the loud audio
    play(loud_audio)

if __name__ == "__main__":
    # image_url = "https://plumsail.com/static/eade2ba6d4a6bf0947c24f462853c95b/e46ed/thumbnail.png"  # Image URL
    image_url = "https://images.squarespace-cdn.com/content/v1/5e8bd6ab7b0c05317fdbd775/1605477117913-7C3B85R8NU6QC580JKNH/Diabetic+Risk+Assessment+Part+4.png"
    image = download_image(image_url)  # Download the image
    extracted_text = extract_text_from_image(image)  # Extract text from the downloaded image
    
    # Translate the extracted text to Nepali
    nepali_text = translate_text(extracted_text, dest_language='ne')
    print("Translated Text in Nepali:")
    print(nepali_text)
    
    # Save the translated text to a file
    with open('translated_text_nepali.txt', 'w') as text_file:
        text_file.write(nepali_text)
        print("Translated text saved as 'translated_text_nepali.txt'.")

    # Speak the translated text in Nepali loudly
    speak_text(nepali_text, language='ne', volume_db=10)  # Increase volume by 10 dB
