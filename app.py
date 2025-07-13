import streamlit as st
import pandas as pd
import joblib
import nltk
import fitz  # PyMuPDF for PDF handling

nltk.download('punkt')

# Load saved model and vectorizer
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Set up Streamlit page
st.set_page_config(page_title="Spam Detector", page_icon="📩")
st.title("📩 SMS Spam & Fraud Detector")
st.write("Detect spam messages using AI - Type or upload CSV/PDF files.")

# --- Section 1: Single Message ---
st.header("🔎 Check a Single Message")
user_input = st.text_area("Enter your SMS message here:")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message.")
    else:
        input_vector = vectorizer.transform([user_input])
        result = model.predict(input_vector)
        if result[0] == 1:
            st.error("🚫 This message is SPAM!")
        else:
            st.success("✅ This message is NOT SPAM.")

# --- Section 2: File Upload (CSV or PDF) ---
st.header("📂 Upload a CSV or PDF File with Messages")

uploaded_file = st.file_uploader("Upload file (.csv or .pdf)", type=['csv', 'pdf'])

if uploaded_file is not None:
    try:
        # --- If CSV file ---
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)

            if 'message' not in data.columns:
                st.error("❌ CSV file must contain a 'message' column.")
            else:
                X_input = vectorizer.transform(data['message'])
                predictions = model.predict(X_input)
                data['Prediction'] = ['Spam' if p == 1 else 'Not Spam' for p in predictions]
                st.success("✅ Predictions complete!")
                st.write(data)

                # Download button
                csv = data.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download CSV Results", data=csv, file_name="csv_predictions.csv", mime='text/csv')

        # --- If PDF file ---
        elif uploaded_file.name.endswith('.pdf'):
            # Read PDF and extract all text
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()

            # Split text line by line as separate messages
            messages = [line.strip() for line in text.split('\n') if line.strip()]
            df_pdf = pd.DataFrame({'message': messages})

            X_input = vectorizer.transform(df_pdf['message'])
            predictions = model.predict(X_input)
            df_pdf['Prediction'] = ['Spam' if p == 1 else 'Not Spam' for p in predictions]
            st.success("✅ Predictions complete from PDF!")
            st.write(df_pdf)

            # Download button
            pdf_csv = df_pdf.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download PDF Results", data=pdf_csv, file_name="pdf_predictions.csv", mime='text/csv')

        else:
            st.error("❌ Unsupported file type. Please upload a .csv or .pdf file.")

    except Exception as e:
        st.error(f"⚠️ Error reading file: {e}")
