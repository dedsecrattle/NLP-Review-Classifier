import streamlit as st
from src.inference import classify_review

st.title("📝 Review Quality Checker (MBO-DeBERTa)")
review = st.text_area("Enter a review:")

if st.button("Classify"):
    category, confidences = classify_review(review)
    st.subheader(f"Predicted Category: {category}")
    st.write("Confidence Scores:")
    st.bar_chart(confidences)
