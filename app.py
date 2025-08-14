import cv2
import pytesseract
import pandas as pd
import numpy as np
from PIL import Image
import re

def preprocess_image(image_path):
    # Load and convert to grayscale
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize for better OCR accuracy
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    return thresh

def extract_text(image):
    # Convert OpenCV image to PIL format
    pil_img = Image.fromarray(image)
    text = pytesseract.image_to_string(pil_img, lang='eng')
    return text

def parse_table(text):
    # Split into lines and filter out empty ones
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Try to detect columns using multiple spaces or tabs
    rows = []
    for line in lines:
        # Split by 2+ spaces or tabs
        columns = re.split(r'\s{2,}|\t+', line)
        rows.append(columns)

    # Normalize row lengths
    max_cols = max(len(row) for row in rows)
    rows = [row + [''] * (max_cols - len(row)) for row in rows]

    # Assume first row is header
    df = pd.DataFrame(rows[1:], columns=rows[0])
    return df

def export_to_excel(df, output_path="output.xlsx"):
    df.to_excel(output_path, index=False, engine="openpyxl")
    print(f"✅ Exported to {output_path}")

def process_image_to_excel(image_path, output_path="output.xlsx"):
    processed = preprocess_image(image_path)
    text = extract_text(processed)
    df = parse_table(text)
    export_to_excel(df, output_path)

# Example usage
process_image_to_excel("your_image.jpeg", "table_output.xlsx")
