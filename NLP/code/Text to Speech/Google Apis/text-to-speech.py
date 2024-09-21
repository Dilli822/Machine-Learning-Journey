import pyttsx3

# Initialize the pyttsx3 engine
engine = pyttsx3.init()

# Try to read the file, if it doesn't exist, use default text
try:
    with open('sample.txt', 'r') as file:
        text = file.read()
except FileNotFoundError:
    # If the file is not found, use this default text
    text = "Hello, welcome to Python text-to-speech conversion!"

# Set properties (optional)
engine.setProperty('rate', 150)    # Speed of speech
engine.setProperty('volume', 1)    # Volume level (0.0 to 1.0)

# Say the text (either from file or default)
engine.say(text)

# Run the speech engine
engine.runAndWait()


