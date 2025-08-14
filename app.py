import streamlit as st
from PIL import Image, ImageOps
import pytesseract
import pandas as pd
from io import BytesIO

def preprocess_image(image):
    image = image.convert("L")  # Grayscale
    image = ImageOps.autocontrast(image)
    image = image.point(lambda x: 0 if x < 128 else 255, '1')  # Binarize
    return image

def parse_table_from_data(ocr_df):
    # Drop rows with no text
    ocr_df = ocr_df[ocr_df.text.notnull() & (ocr_df.text.str.strip() != "")]
    ocr_df = ocr_df.reset_index(drop=True)

    # Group by line number
    lines = ocr_df.groupby("line_num")

    rows = []
    for _, line in lines:
        words = line.sort_values("left")["text"].tolist()
        rows.append(words)

    if not rows or len(rows) < 2:
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
    ocr_df = pytesseract.image_to_data(processed, lang="eng", output_type=pytesseract.Output.DATAFRAME)

    df = parse_table_from_data(ocr_df)

    if df.empty:
        st.warning("⚠️ Could not detect a table structure. Try a clearer image.")
        st.text_area("Raw OCR Output", "\n".join(ocr_df["text"].dropna().tolist()), height=200)
    else:
        st.subheader("Extracted Table")
        st.dataframe(df)

        csv_data = convert_df_to_csv(df)
        st.download_button("Download as CSV", data=csv_data, file_name="table.csv", mime="text/csv")
