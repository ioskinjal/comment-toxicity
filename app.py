import streamlit as st
import joblib
import pandas as pd
import os

st.title("🛡️ Comment Toxicity Detector")

# Load model & vectorizer
model_path = "models/toxicity_model.pkl"
vectorizer_path = "models/vectorizer.pkl"

if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    # Single text input
    user_input = st.text_area("Enter a comment:")

    if st.button("Predict"):
        if user_input.strip():
            X = vectorizer.transform([user_input])
            pred = model.predict(X)[0]
            st.write("Prediction:", "⚠️ Toxic" if pred == 1 else "✅ Not Toxic")
        else:
            st.warning("Please enter a comment.")

    # Bulk CSV upload
    uploaded_file = st.file_uploader("Upload CSV for bulk prediction", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "comment_text" in df.columns:
            df['prediction'] = model.predict(vectorizer.transform(df['comment_text']))
            st.write(df.head())
        else:
            st.error("CSV must have a 'comment_text' column.")
else:
    st.error("❌ Model not found! Please run `python train_model.py` first.")