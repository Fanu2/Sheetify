import streamlit as st
from PIL import Image, ImageOps
import pytesseract
import pandas as pd
import re
from io import BytesIO

def preprocess_image(image):
    image = image.convert("L")  # Grayscale
    image = ImageOps.autocontrast(image)
    return image

def parse_table(text):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    rows = [re.split(r"\s{2,}|\t+", line) for line in lines]
    max_cols = max(len(row) for row in rows)
    rows = [row + [""] * (max_cols - len(row)) for row in rows]
    return pd.DataFrame(rows[1:], columns=rows[0])

def convert_df_to_excel(df):
    output = BytesIO()
    df.to_excel(output, index=False, engine="openpyxl")
    output.seek(0)
    return output

# Streamlit UI
st.title("📄 OCR Table Extractor")
uploaded_file = st.file_uploader("Upload an image of a table", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    processed = preprocess_image(image)
    text = pytesseract.image_to_string(processed, lang="eng")
    df = parse_table(text)

    st.subheader("Extracted Table")
    st.dataframe(df)

    excel_data = convert_df_to_excel(df)
    st.download_button("Download as Excel", data=excel_data, file_name="table.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
