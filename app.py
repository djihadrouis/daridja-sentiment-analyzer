!pip install dash dash-bootstrap-components joblib

import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import joblib
import os

# ======== Charger modèle et tokenizer ========
MODEL_PATH = r"C:\Djihad\M2\NLP\daridja_sentiment_app\sentiment_lstm_model.h5"
TOKENIZER_PATH = r"C:\Djihad\M2\NLP\daridja_sentiment_app\tokenizer.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
tokenizer = joblib.load(TOKENIZER_PATH)
max_len = 100  # adapte selon ton preprocessing

# ======== Initialiser l’app Dash ========
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Daridja Sentiment Analyzer"

# ======== Layout ========
app.layout = dbc.Container([
    html.H1("🧠 Daridja Sentiment Analyzer", className="text-center my-4 text-primary"),

    dbc.Row([
        dbc.Col([
            dbc.Textarea(id="input-text", placeholder="Écris ton commentaire en Daridja...", rows=5),
            html.Br(),
            dbc.Button("Prédire", id="predict-btn", color="primary", className="w-100"),
            html.Br(), html.Br(),
            html.H5("Résultat :", className="text-info"),
            html.Div(id="output-label", className="fw-bold fs-4"),
            html.Div(id="confidence", className="text-muted mt-2")
        ], width=6)
    ], justify="center"),
], fluid=True)

# ======== Callback ========
@app.callback(
    [Output("output-label", "children"),
     Output("confidence", "children")],
    [Input("predict-btn", "n_clicks")],
    [State("input-text", "value")]
)
def predict_sentiment(n_clicks, text):
    if not n_clicks or not text or not text.strip():
        return "", ""

    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')

    preds = model.predict(padded)
    arr = np.asarray(preds).ravel()

    # Gérer sigmoïde (taille 1) ou softmax (taille >=2)
    if arr.size == 1:
        p_pos = float(arr[0])
        p_neg = 1.0 - p_pos
    elif arr.size >= 2:
        # supposé [neg, pos] ou [pos, neg] selon l'entraînement
        # on suppose l'ordre [neg, pos] comme précédemment
        p_neg = float(arr[0])
        p_pos = float(arr[1])
    else:
        p_pos = 0.0
        p_neg = 0.0

    predicted = "Positive" if p_pos >= p_neg else "Negative"

    confidence_text = f"Confiance : Positive = {p_pos*100:.2f}%, Negative = {p_neg*100:.2f}%"
    return f"🧾 Sentiment prédit : {predicted}", confidence_text

# ======== Lancer app ========
if __name__ == "__main__":
    app.run(debug=True, port=8050)
