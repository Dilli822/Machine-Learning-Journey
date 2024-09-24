import pyttsx3
import os
import random
from PyPDF2 import PdfReader
from docx import Document

# Initialize the pyttsx3 engine
engine = pyttsx3.init()

def read_text_from_file(file_path):
    """Read text from a file based on its extension."""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.txt':
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    elif ext == '.pdf':
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
    elif ext == '.docx':
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    else:
        return None

# Try to read from a file (text, PDF, or Word), or use default text
file_path = 'sm.txt'  # Change this to your desired file path
text = "Please Upload the file!"

try:
    text = read_text_from_file(file_path)
    if not text:
        raise ValueError("Unsupported file format or empty file.")
except (FileNotFoundError, ValueError) as e:
    text = "Hello, welcome to Python text-to-speech conversion!"

# Set properties (optional)
engine.setProperty('rate', 200)    # Speed of speech
engine.setProperty('volume', 1)    # Volume level (0.0 to 1.0)

# List available voices and filter for English voices
voices = engine.getProperty('voices')
english_voices = [voice for voice in voices if 'en' in voice.languages]

# Randomly select an English voice
if english_voices:
    random_voice = random.choice(english_voices)
    engine.setProperty('voice', random_voice.id)
else:
    print("No English voice found. Using default voice.")

# Say the text (either from file or default)
engine.say(text)

# Run the speech engine
engine.runAndWait()
