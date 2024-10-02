import cv2
import pytesseract
from PIL import Image

# If tesseract is not in your PATH, specify the location
# For example, on Windows:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the image
image_path = 'scanned_image.png'  # Replace with your image file
img = cv2.imread(image_path)

# Check if the image was successfully loaded
if img is None:
    print(f"Error: Could not open or read the image {image_path}")
    exit()

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Optional: Apply thresholding if the text is unclear (e.g., noise or low contrast)
# _, thresholded = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# You can pass either the grayscale or thresholded image to Tesseract for OCR
# In this case, we'll just use the grayscale image directly

# Use pytesseract to extract text from the image
extracted_text = pytesseract.image_to_string(gray)

# Display the extracted text
print("Extracted Text:")
print(extracted_text)
