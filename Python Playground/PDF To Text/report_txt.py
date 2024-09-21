import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import os

def pdf_to_text(pdf_path, txt_path):
    extracted_text = []

    try:
        # Attempt to extract text with pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                try:
                    text = page.extract_text()
                    if text and text.strip():  # Check if extracted text is not empty
                        extracted_text.append(text)
                    else:
                        # If no text, use OCR on the page image
                        print(f"Page {i + 1} has no readable text, using OCR.")
                        images = convert_from_path(pdf_path, first_page=i + 1, last_page=i + 1)
                        for image in images:
                            # Perform OCR on the image
                            ocr_text = pytesseract.image_to_string(image)
                            if ocr_text.strip():  # Check if OCR text is not empty
                                extracted_text.append(ocr_text)
                            else:
                                print(f"OCR could not read text on Page {i + 1}.")
                except Exception as e:
                    print(f"Error processing page {i + 1}: {e}")

    except FileNotFoundError:
        print(f"File not found: {pdf_path}")
        return
    except Exception as e:
        print(f"Error opening PDF file: {e}")
        return

    # Write the extracted text to a .txt file
    try:
        with open(txt_path, 'w', encoding='utf-8') as text_file:
            for page_text in extracted_text:
                text_file.write(page_text)
                text_file.write("\n\n")  # Add space between pages
        print(f"Extraction complete. Text saved to {txt_path}")
    except Exception as e:
        print(f"Error writing to output file: {e}")

# Example usage
pdf_to_text('medical_report.pdf', 'output.txt')
