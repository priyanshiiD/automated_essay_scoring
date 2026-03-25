import pickle
import re
import streamlit as st
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from scipy.sparse import hstack
import nltk

# ================================
# NLTK SETUP
# ================================
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="Automated Essay Scoring System",
    page_icon="📝",
)

# ================================
# LOAD MODEL FILES
# ================================
model = pickle.load(open("essay_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ================================
# SAMPLE ESSAYS
# ================================
SAMPLE_ESSAYS = {
    "School Uniforms": "School uniforms should be required in all public schools because they help students focus on learning instead of fashion. When everyone wears the same clothes, social pressure is reduced.",
    
    "Online Learning": "Online learning has made education more flexible and accessible. Students can learn at their own pace and access resources anytime, but it also creates challenges like distractions.",
    
    "Social Media Impact": "Social media has both positive and negative effects. It helps people connect but can also lead to distraction and mental health issues if overused."
}

# ================================
# TEXT CLEANING
# ================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# ================================
# FEATURE EXTRACTION
# ================================
def get_essay_stats(text):
    word_count = len(text.split())
    sentence_count = len(sent_tokenize(text)) if text.strip() else 0
    avg_sentence_length = word_count / max(sentence_count, 1)

    return word_count, sentence_count, avg_sentence_length

# ================================
# PREDICTION FUNCTION
# ================================
def predict(text):
    cleaned = clean_text(text)

    word_count, sentence_count, avg_sentence_length = get_essay_stats(text)

    struct = [[word_count, sentence_count, avg_sentence_length]]
    struct_scaled = scaler.transform(struct)

    tfidf = vectorizer.transform([cleaned])

    final_features = hstack([tfidf, struct_scaled])

    score = model.predict(final_features)[0]

    return score, word_count, sentence_count, avg_sentence_length

# ================================
# FEEDBACK FUNCTION (FIXED)
# ================================
def get_feedback(score):
    if score < 2.5:
        return "Needs Improvement", "Try improving clarity, grammar, and structure."
    elif score < 4:
        return "Average", "Good attempt. Add stronger arguments and examples."
    else:
        return "Strong", "Well-structured essay with good clarity."

# ================================
# UI
# ================================
st.title("📝 Automated Essay Scoring System")

st.write("This system predicts essay scores using machine learning based on writing patterns and structure.")

# Sidebar
st.sidebar.header("About")
st.sidebar.write("Model trained on ASAP dataset")
st.sidebar.write("Uses TF-IDF + structural features")

# Sample selection
sample_choice = st.selectbox(
    "Try a sample essay",
    ["None"] + list(SAMPLE_ESSAYS.keys())
)

if st.button("Load Sample"):
    if sample_choice != "None":
        st.session_state["essay"] = SAMPLE_ESSAYS[sample_choice]

# Text input
essay = st.text_area(
    "Enter your essay",
    height=250,
    key="essay"
)

# Show live stats
if essay.strip():
    wc, sc, avg_len = get_essay_stats(essay)
    col1, col2, col3 = st.columns(3)
    col1.metric("Words", wc)
    col2.metric("Sentences", sc)
    col3.metric("Avg Sentence Length", round(avg_len, 1))

# Predict button
if st.button("Predict Score"):

    if essay.strip() == "":
        st.warning("Please enter an essay.")
    else:
        with st.spinner("Analyzing essay..."):
            score, wc, sc, avg_len = predict(essay)

        # Optional scaling for demo (can remove if you want)
        adjusted_score = min(6, max(1, score * 1.3))

        quality, feedback = get_feedback(adjusted_score)

        st.success(f"Predicted Score: {round(adjusted_score, 2)}")
        st.info(f"Quality: {quality}")

        st.write("**Feedback:**", feedback)

        col1, col2, col3 = st.columns(3)
        col1.metric("Words", wc)
        col2.metric("Sentences", sc)
        col3.metric("Avg Sentence Length", round(avg_len, 1))