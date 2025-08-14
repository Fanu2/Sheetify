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

def deduplicate_columns(columns):
    seen = {}
    new_cols = []
    for col in columns:
        if col not in seen:
            seen[col] = 1
            new_cols.append(col)
        else:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
    return new_cols

def parse_table_from_data(ocr_df):
    ocr_df = ocr_df[ocr_df.text.notnull() & (ocr_df.text.str.strip() != "")]
    ocr_df = ocr_df.reset_index(drop=True)

    lines = ocr_df.groupby("line_num")
    rows = []
    for _, line in lines:
        words = line.sort_values("left")["text"].tolist()
        rows.append(words)

    if not rows or len(rows) < 2:
        return pd.DataFrame()

    max_cols = max(len(row) for row in rows)
    rows = [row + [""] * (max_cols - len(row)) for row in rows]

    headers = deduplicate_columns(rows[0])
    df = pd.DataFrame(rows[1:], columns=headers)
    return df


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
