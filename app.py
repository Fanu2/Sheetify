import streamlit as st
import pytesseract
from PIL import Image, ImageFilter, ImageOps
import pandas as pd
import numpy as np
# from pyexcel_ods import save_data
from collections import OrderedDict
import io
import xlsxwriter

# Optional: Set Tesseract path if needed
# pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# 🚀 App Configuration
st.set_page_config(page_title="Image to Spreadsheet", layout="wide")
st.title("🖼️ Image to Spreadsheet Converter")
st.markdown("Convert tabular data from scanned images into CSV, Excel, or LibreOffice `.ods` format.")

# 📌 Image Preprocessing
def preprocess_image(pil_image, threshold=180):
    gray = pil_image.convert("L")
    blurred = gray.filter(ImageFilter.GaussianBlur(radius=1))
    enhanced = ImageOps.autocontrast(blurred)
    np_img = np.array(enhanced)
    binary = np.where(np_img > threshold, 255, 0).astype(np.uint8)
    return Image.fromarray(binary)

# 📌 Table Extraction
def extract_table_data(image):
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DATAFRAME)
    ocr_data = ocr_data.dropna(subset=['text'])
    rows, current_line = [], []
    last_line_num = -1

    for _, row in ocr_data.iterrows():
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

# 🧠 Header Detection
def detect_header(df):
    if df.shape[0] > 1 and all(str(cell).isalpha() for cell in df.iloc[0]):
        df.columns = df.iloc[0]
        df = df.drop(index=0).reset_index(drop=True)
    return df

# 📤 Excel Export
def export_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()

# 📥 Multi-Image Upload
uploaded_files = st.file_uploader("Upload one or more images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    threshold = st.slider("🧪 Binarization Threshold", min_value=0, max_value=255, value=180)

    for uploaded_file in uploaded_files:
        st.markdown(f"---\n### 📷 Processing: `{uploaded_file.name}`")
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)

        processed_image = preprocess_image(image, threshold)
        st.image(processed_image, caption="Preprocessed Image", use_column_width=True)

        raw_text = pytesseract.image_to_string(processed_image)
        st.text_area("📝 Extracted Text", raw_text, height=150)

        df = extract_table_data(processed_image)
        df = detect_header(df)

        st.subheader("📊 Parsed Table")
        edited_df = st.experimental_data_editor(df, num_rows="dynamic")

        export_format = st.selectbox(f"Choose export format for `{uploaded_file.name}`", ["CSV", "Excel", "ODS"])

        if st.button(f"Download `{uploaded_file.name}`"):
            if export_format == "CSV":
                csv = edited_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download CSV", csv, f"{uploaded_file.name}.csv", "text/csv")
            elif export_format == "Excel":
                excel_data = export_to_excel(edited_df)
                st.download_button("Download Excel", excel_data, f"{uploaded_file.name}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            else:
                data = OrderedDict()
                data["Sheet1"] = [edited_df.columns.tolist()] + edited_df.values.tolist()
                buffer = io.BytesIO()
                save_data(buffer, data)
                st.download_button("Download ODS", buffer.getvalue(), f"{uploaded_file.name}.ods", "application/vnd.oasis.opendocument.spreadsheet")
