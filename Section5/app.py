import streamlit as st
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from transformers import pipeline

st.title("Fine Tuning BERT for Twitter Multi Class Sentiment Classification")

classifier = pipeline("text-classification", model="bert-base-uncased-sentiment", device=device)

text = st.text_area("Enter the Text to classify")

if st.button("Predict"):
    result = classifier(text)
    st.write(f"**Prediction:** {result[0]['label']}")
    st.markdown(f"The score is **{result[0]['score']}**.")

