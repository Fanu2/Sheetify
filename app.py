import streamlit as st
import pytesseract
from PIL import Image
import pandas as pd
import cv2
import numpy as np
from pyexcel_ods import save_data
from collections import OrderedDict
import io

st.set_page_config(page_title="Image to Spreadsheet", layout="wide")
st.title("🖼️ Image to Spreadsheet Converter")
st.markdown("Convert tabular data from scanned images into CSV or LibreOffice `.ods` format.")

# 📌 Image Preprocessing
def preprocess_image(pil_image):
    img = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 10)
    return Image.fromarray(thresh)

# 📌 Table Extraction
def extract_table_data(image):
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DATAFRAME)
    data = data.dropna(subset=['text'])
    rows = []
    current_line = []
    last_line_num = -1

    for _, row in data.iterrows():
        if row['line_num'] != last_line_num:
            if current_line:
                rows.append(current_line)
            current_line = [row['text']]
            last_line_num = row['line_num']
        else:
            current_line.append(row['text'])

    if current_line:
        rows.append(current_line)

    return pd.DataFrame(rows)

# 📥 Upload
uploaded_file = st.file_uploader("Upload an image with tabular data", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_column_width=True)

    # Preprocess and OCR
    processed_image = preprocess_image(image)
    st.image(processed_image, caption="Preprocessed Image", use_column_width=True)

    raw_text = pytesseract.image_to_string(processed_image)
    st.text_area("📝 Extracted Text", raw_text, height=200)

    df = extract_table_data(processed_image)
    st.subheader("📊 Parsed Table")
    st.dataframe(df)

    # Export
    export_format = st.selectbox("Choose export format", ["CSV", "ODS"])

    if st.button("Download"):
        if export_format == "CSV":
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv, "converted.csv", "text/csv")
        else:
            data = OrderedDict()
            data.update({"Sheet1": [df.columns.tolist()] + df.values.tolist()})
            buffer = io.BytesIO()
            save_data(buffer, data)
            st.download_button("Download ODS", buffer.getvalue(), "converted.ods", "application/vnd.oasis.opendocument.spreadsheet")
