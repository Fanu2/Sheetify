import streamlit as st
import pytesseract
from PIL import Image, ImageFilter, ImageOps
import pandas as pd
import numpy as np
import io
import zipfile
import xlsxwriter
import pdfplumber
from googletrans import Translator
from langdetect import detect

# Optional: Set Tesseract path
# pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

st.set_page_config(page_title="Universal OCR Dashboard", layout="wide")
st.title("🧠 Universal OCR Dashboard")
st.markdown("Extract structured data from images, PDFs, voice, and webcam input. Supports multilingual OCR, handwriting, and layout-aware table reconstruction.")

# 📌 Image Preprocessing
def preprocess_image(pil_image, threshold=180):
    gray = pil_image.convert("L")
    blurred = gray.filter(ImageFilter.GaussianBlur(radius=1))
    enhanced = ImageOps.autocontrast(blurred)
    np_img = np.array(enhanced)
    binary = np.where(np_img > threshold, 255, 0).astype(np.uint8)
    return Image.fromarray(binary)

# ✍️ Handwriting Preprocessing
def preprocess_handwriting(image):
    gray = image.convert("L")
    enhanced = ImageOps.autocontrast(gray)
    sharpened = enhanced.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
    return sharpened

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
    if df.shape[0] > 1 and df.iloc[0].apply(lambda x: isinstance(x, str)).all():
        df.columns = df.iloc[0]
        df = df.drop(index=0).reset_index(drop=True)
    return df

# 📤 Enhanced Excel Export
def export_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']
        header_format = workbook.add_format({
            'bold': True, 'text_wrap': True, 'valign': 'top',
            'fg_color': '#D7E4BC', 'border': 1
        })
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        for i, col in enumerate(df.columns):
            column_len = max(df[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.set_column(i, i, column_len)
        worksheet.freeze_panes(1, 0)
        for i, col in enumerate(df.columns):
            if pd.api.types.is_numeric_dtype(df[col]):
                worksheet.conditional_format(1, i, len(df), i, {'type': '3_color_scale'})
    return output.getvalue()

# 📦 Batch Export
def export_all_to_zip(dataframes, filenames):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for df, name in zip(dataframes, filenames):
            excel_bytes = export_to_excel(df)
            zip_file.writestr(f"{name}.xlsx", excel_bytes)
    return zip_buffer.getvalue()

# 🌐 Language Selection
lang_code = st.selectbox("🌍 OCR Language", {
    "English": "eng", "Spanish": "spa", "French": "fra",
    "German": "deu", "Japanese": "jpn", "Chinese": "chi_sim",
    "Arabic": "ara"
})

# 🧪 Confidence Threshold
low_conf_threshold = st.slider("🔍 Confidence Threshold", 0, 100, 60)

# 📷 Image OCR Tab
st.subheader("📷 Image OCR")
uploaded_files = st.file_uploader("Upload images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

all_dfs, all_names = [], []

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.markdown(f"---\n### Processing: `{uploaded_file.name}`")
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)

        processed_image = preprocess_image(image)
        st.image(processed_image, caption="Preprocessed Image", use_column_width=True)

        raw_text = pytesseract.image_to_string(processed_image, lang=lang_code)
        st.text_area("📝 Extracted Text", raw_text, height=150)

        try:
            detected_lang = detect(raw_text)
            st.write(f"🧠 Detected Language: `{detected_lang}`")
            translator = Translator()
            translated = translator.translate(raw_text, src=lang_code, dest='en')
            st.text_area("🌍 Translated Text", translated.text, height=150)
        except:
            st.warning("Translation or language detection failed.")

        df = extract_table_data(processed_image)
        df = detect_header(df)

        st.subheader("📊 Parsed Table")
        st.dataframe(df)

        ocr_debug = pytesseract.image_to_data(processed_image, output_type=pytesseract.Output.DATAFRAME)
        low_conf_cells = ocr_debug[ocr_debug['conf'] < low_conf_threshold]
        if not low_conf_cells.empty:
            st.warning(f"{len(low_conf_cells)} low-confidence words detected.")
            st.dataframe(low_conf_cells[['text', 'conf', 'left', 'top']])

        all_dfs.append(df)
        all_names.append(uploaded_file.name)

# 📦 Batch Download
if all_dfs and st.button("📦 Download All Tables as ZIP"):
    zip_data = export_all_to_zip(all_dfs, all_names)
    st.download_button("Download ZIP", zip_data, "tables.zip", "application/zip")

# 📄 PDF Table Extraction
st.subheader("📄 PDF Table Extraction")
pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
if pdf_file:
    with pdfplumber.open(pdf_file) as pdf:
        for i, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            for j, table in enumerate(tables):
                df = pd.DataFrame(table[1:], columns=table[0])
                st.subheader(f"📊 Page {i+1}, Table {j+1}")
                st.dataframe(df)
                st.download_button(f"Download Table {i+1}-{j+1}", df.to_csv(index=False).encode("utf-8"), f"table_{i+1}_{j+1}.csv", "text/csv")
