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
    rows = []
    for line in lines:
        # Split by 2+ spaces or tabs
        columns = re.split(r"\s{2,}|\t+", line)
        if len(columns) > 1:
            rows.append(columns)
    if not rows:
        return pd.DataFrame()
    max_cols = max(len(row) for row in rows)
    rows = [row + [""] * (max_cols - len(row)) for row in rows]
    df = pd.DataFrame(rows[1:], columns=rows[0])
    return df

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")

# Streamlit UI
st.title("📄 OCR Table to CSV")
uploaded_file = st.file_uploader("Upload an image of a table", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    processed = preprocess_image(image)
    text = pytesseract.image_to_string(processed, lang="eng")
    df = parse_table(text)

    if df.empty:
        st.warning("⚠️ Could not detect a table structure. Try a clearer image.")
    else:
        st.subheader("Extracted Table")
        st.dataframe(df)

        csv_data = convert_df_to_csv(df)
        st.download_button("Download as CSV", data=csv_data, file_name="table.csv", mime="text/csv")
