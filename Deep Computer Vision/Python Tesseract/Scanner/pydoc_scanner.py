import cv2
import pytesseract
from PIL import Image

# Load the image
image_path = 'captured_image.jpg'
img = cv2.imread(image_path)

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply thresholding to make the text clearer
# You can adjust the thresholding method based on your image quality
_, thresholded = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)

# Save the processed image
scanned_image_path = 'scanned_image.png'
cv2.imwrite(scanned_image_path, thresholded)

# Optionally, use PyTesseract for OCR (to extract text)
# Make sure to have Tesseract installed (check tesseract documentation for installation)
text = pytesseract.image_to_string(thresholded)

# Display the text from OCR (optional)
print("Extracted Text:")
print(text)

# Show the original and processed images side by side (optional)
cv2.imshow('Original Image', img)
cv2.imshow('Processed (Scanned) Image', thresholded)
cv2.waitKey(0)
cv2.destroyAllWindows()
