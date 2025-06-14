import streamlit as st
import pandas as pd
import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Judul aplikasi
st.title("📩 Spam Message Detector")

# Cek apakah model sudah ada
MODEL_FILE = "model.pkl"
VECTORIZER_FILE = "vectorizer.pkl"

# Jika belum ada, latih model
if not os.path.exists(MODEL_FILE) or not os.path.exists(VECTORIZER_FILE):
    st.info("Training model...")

    # Load data
    df = pd.read_csv("SMSSpamCollection", sep='\t', header=None, names=['label', 'message'])

    # Preprocessing dan pelatihan
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['message'])
    y = df['label']

    model = MultinomialNB()
    model.fit(X, y)

    # Simpan model
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    with open(VECTORIZER_FILE, "wb") as f:
        pickle.dump(vectorizer, f)

    st.success("Model trained successfully!")

# Load model yang sudah ada
with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_FILE, "rb") as f:
    vectorizer = pickle.load(f)

# Input dari user
st.subheader("Cek apakah pesan termasuk spam atau bukan:")
user_input = st.text_area("Masukkan pesan di sini...")

if st.button("🔍 Deteksi"):
    if user_input.strip() == "":
        st.warning("Pesan tidak boleh kosong.")
    else:
        user_vector = vectorizer.transform([user_input])
        prediction = model.predict(user_vector)

        if prediction[0] == "spam":
            st.error("🚫 Pesan ini adalah SPAM.")
        else:
            st.success("✅ Pesan ini BUKAN spam.")
