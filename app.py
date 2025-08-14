from PIL import Image, ImageOps
import pytesseract
import pandas as pd
import re
import os

def preprocess_image(image_path):
    """Simple preprocessing using Pillow for better OCR accuracy."""
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    img = ImageOps.invert(img)  # Optional: invert for better contrast
    img = ImageOps.autocontrast(img)  # Normalize contrast
    return img

def extract_text(image):
    """Run OCR using pytesseract."""
    return pytesseract.image_to_string(image, lang="eng")

def parse_table(text):
    """Parse OCR text into structured rows."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    rows = []
    for line in lines:
        columns = re.split(r"\s{2,}|\t+", line)
        rows.append(columns)
    max_cols = max(len(row) for row in rows)
    rows = [row + [""] * (max_cols - len(row)) for row in rows]
    df = pd.DataFrame(rows[1:], columns=rows[0])
    return df

def export_to_excel(df, output_path="output.xlsx"):
    """Export DataFrame to Excel."""
    df.to_excel(output_path, index=False, engine="openpyxl")
    print(f"✅ Exported to {output_path}")

def process_image_to_excel(image_path, output_path="output.xlsx"):
    """Full pipeline for Vercel-friendly OCR."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = preprocess_image(image_path)
    text = extract_text(image)
    df = parse_table(text)
    export_to_excel(df, output_path)

# Example usage
process_image_to_excel("your_image.jpeg", "table_output.xlsx")
