import joblib
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Load Model and Tokenizer ---
@st.cache_resource
def load_assets():
    model = load_model("model.h5")
    tokenizer = joblib.load("tokenizer.pkl")
    return model, tokenizer

model, tokenizer = load_assets()

# --- UI ---
st.set_page_config(page_title="CineSentiment 🎬", page_icon="🎬", layout="centered")

st.title("🎬 CineSentiment")
st.write("Analyze the **sentiment** of any movie review using a trained LSTM model.")

review = st.text_area("📝 Enter your movie review:")

if st.button("Analyze Sentiment"):
    if review.strip():
        seq = tokenizer.texts_to_sequences([review])
        padded = pad_sequences(seq, maxlen=200)   # use the same maxlen from training
        prediction = model.predict(padded)[0][0]
        sentiment = "😊 Positive" if prediction > 0.5 else "😞 Negative"
        message = "Glad you loved it!" if prediction > 0.5 else "Sorry to hear."

        st.success(f"**Sentiment:** {sentiment}")
        st.write(f"💬 {message}")
        st.caption(f"Confidence: {prediction:.2f}")
    else:
        st.warning("Please enter a review before analyzing.")
