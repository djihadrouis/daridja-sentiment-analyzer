import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import os
import numpy as np
import plotly.graph_objects as go

# ====== Paths (relative) ======
MODEL_PATH = "models/sentiment_lstm_model.h5"
TOKENIZER_PATH = "models/tokenizer.pkl"

# ====== Load model and tokenizer ======
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource
def load_tokenizer():
    return joblib.load(TOKENIZER_PATH)

model = load_model()
tokenizer = load_tokenizer()
max_len = 100

# ====== Page setup ======
st.set_page_config(page_title="Daridja Sentiment Analyzer", layout="wide")
st.title("🧠 Daridja Sentiment Analyzer")
st.write("Analyse le sentiment d'un commentaire écrit en Daridja.")

# ====== Input ======
text_input = st.text_area("Écris ton commentaire en Daridja...", height=150)

# ====== Predict button ======
if st.button("Prédire") and text_input.strip():
    # Preprocessing
    seq = tokenizer.texts_to_sequences([text_input])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')

    # Predict
    preds = model.predict(padded, verbose=0)
    arr = np.asarray(preds).ravel()

    if arr.size == 1:
        p_pos = float(arr[0])
        p_neg = 1.0 - p_pos
    elif arr.size >= 2:
        p_neg = float(arr[0])
        p_pos = float(arr[1])
    else:
        p_pos, p_neg = 0.0, 0.0

    predicted = "Positive" if p_pos >= p_neg else "Negative"
    st.subheader(f"🧾 Sentiment prédit : {predicted}")
    st.write(f"Confiance : Positive = {p_pos*100:.2f}%, Negative = {p_neg*100:.2f}%")

    # ====== Bar chart ======
    bar_chart = go.Figure(go.Bar(
        x=['Négatif', 'Positif'],
        y=[p_neg, p_pos],
        marker_color=['#dc3545', '#17a2b8']
    ))
    bar_chart.update_layout(
        title='Distribution des Sentiments',
        xaxis_title='Sentiment',
        yaxis_title='Probabilité',
        yaxis=dict(range=[0, 1]),
        height=400,
        margin=dict(l=40, r=40, t=40, b=40)
    )

    # ====== Pie chart ======
    pie_chart = go.Figure(go.Pie(
        labels=['Négatif', 'Positif'],
        values=[p_neg, p_pos],
        hole=0.3,
        marker_colors=['#dc3545', '#17a2b8']
    ))
    pie_chart.update_layout(
        title='Proportion des Sentiments',
        height=400,
        margin=dict(l=40, r=40, t=40, b=40)
    )

    # Display charts side by side
    col1, col2 = st.columns(2)
    col1.plotly_chart(bar_chart, use_container_width=True)
    col2.plotly_chart(pie_chart, use_container_width=True)

# ====== Sidebar instructions ======
st.sidebar.title("ℹ️ Instructions")
st.sidebar.write("""
1. Écris ton commentaire en Daridja dans la zone de texte.  
2. Clique sur **Prédire** pour analyser le sentiment.  
3. Visualise la distribution et la proportion des sentiments.  
""")
