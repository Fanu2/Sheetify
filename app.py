import cv2
import pytesseract
import pandas as pd
import numpy as np
from PIL import Image
import re
import os

def preprocess_image(image_path):
    """Load and clean up the image for better OCR accuracy."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    return thresh

def extract_text(image):
    """Run OCR on the preprocessed image."""
    pil_img = Image.fromarray(image)
    text = pytesseract.image_to_string(pil_img, lang='eng')
    return text

def parse_table(text):
    """Convert OCR text into structured rows and columns."""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    rows = []
    for line in lines:
        columns = re.split(r'\s{2,}|\t+', line)
        rows.append(columns)
    max_cols = max(len(row) for row in rows)
    rows = [row + [''] * (max_cols - len(row)) for row in rows]
    df = pd.DataFrame(rows[1:], columns=rows[0])
    return df

def export_to_excel(df, output_path="output.xlsx"):
    """Save the DataFrame to an Excel file."""
    df.to_excel(output_path, index=False, engine="openpyxl")
    print(f"✅ Exported to {output_path}")

def process_image_to_excel(image_path, output_path="output.xlsx"):
    """Full pipeline: preprocess, OCR, parse, and export."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    processed = preprocess_image(image_path)
    text = extract_text(processed)
    df = parse_table(text)
    export_to_excel(df, output_path)

# Example usage
process_image_to_excel("your_image.jpeg", "table_output.xlsx")
