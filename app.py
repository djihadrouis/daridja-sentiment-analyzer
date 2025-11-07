import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import joblib
import os

# ======== Load model and tokenizer ========
MODEL_PATH = "models/sentiment_lstm_model.h5"
TOKENIZER_PATH = "models/tokenizer.pkl"

# Verify files exist
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model not found: {MODEL_PATH}")
    st.stop()
if not os.path.exists(TOKENIZER_PATH):
    st.error(f"❌ Tokenizer not found: {TOKENIZER_PATH}")
    st.stop()

# Load model and tokenizer
model = tf.keras.models.load_model(MODEL_PATH)
tokenizer = joblib.load(TOKENIZER_PATH)
max_len = 100  # adapt based on preprocessing

# ======== Streamlit Layout ========
st.set_page_config(page_title="Daridja Sentiment Analyzer", page_icon="🧠")

# --- ✅ Cover image ---
cover = Image.open("daridja_sentiment_couvertur.png")
st.image(cover, use_column_width=True)

# --- Title and description ---
st.title("🧠 Daridja Sentiment Analyzer")
st.write("Analyze the sentiment of comments written in **Daridja 🇩🇿** using a deep learning LSTM model.")

# ======== Input Area ========
st.subheader("📝 Enter your text in Daridja:")
user_text = st.text_area("Write here...", height=150)

if st.button("🔍 Analyze Sentiment"):
    if not user_text.strip():
        st.warning("⚠️ Please enter some text before analyzing.")
    else:
        # Preprocess text
        seq = tokenizer.texts_to_sequences([user_text])
        padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')

        # Prediction
        preds = model.predict(padded)
        arr = np.asarray(preds).ravel()

        # Handle sigmoid or softmax outputs
        if arr.size == 1:
            p_pos = float(arr[0])
            p_neg = 1.0 - p_pos
        else:
            p_neg = float(arr[0])
            p_pos = float(arr[1])

        predicted = "Positive 😊" if p_pos >= p_neg else "Negative 😠"

        # Display results
        st.success(f"**Predicted Sentiment:** {predicted}")
        st.write(f"Confidence — Positive: `{p_pos*100:.2f}%`, Negative: `{p_neg*100:.2f}%`")

# ======== Footer ========
st.markdown("---")
st.markdown("Made with ❤️ by **Djihad Rouis** — LSTM Model for Daridja 🇩🇿")
