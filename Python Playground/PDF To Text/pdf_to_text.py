import pdfplumber

def pdf_to_text(pdf_path, txt_path):
    with pdfplumber.open(pdf_path) as pdf:
        with open(txt_path, 'w', encoding='utf-8') as text_file:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_file.write(text)
                    text_file.write("\n\n")  # Adds a newline between pages for better readability

# Replace 'input.pdf' with your PDF file and 'output.txt' with your desired output text file
pdf_to_text('1725434935.pdf', 'output.txt')
