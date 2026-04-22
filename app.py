import streamlit as st
import pickle

# laad model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

st.title("📰 Fake News Detector")

text = st.text_area("Plak hier een artikel of zin:")

if st.button("Analyseer"):
    if text.strip() == "":
        st.warning("Voer eerst tekst in")
    else:
        vec = tfidf.transform([text])
        pred = model.predict(vec)[0]

        if pred == 0:
            st.error("❌ FAKE NEWS")
        else:
            st.success("✅ REAL NEWS")