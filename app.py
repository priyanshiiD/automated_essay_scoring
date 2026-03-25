import pickle
import re
import streamlit as st
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
    "School Uniforms": "School uniforms should be required because they reduce distractions and improve discipline among students.",
    "Online Learning": "Online learning provides flexibility but also introduces challenges like lack of interaction and distractions.",
    "Social Media Impact": "Social media connects people but can negatively affect mental health if overused."
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
def get_stats(text):
    word_count = len(text.split())
    sentence_count = len(sent_tokenize(text)) if text.strip() else 0
    avg_sentence_length = word_count / max(sentence_count, 1)
    return word_count, sentence_count, avg_sentence_length

# ================================
# PREDICTION FUNCTION
# ================================
def predict(text):
    cleaned = clean_text(text)

    word_count, sentence_count, avg_sentence_length = get_stats(text)

    struct = [[word_count, sentence_count, avg_sentence_length]]
    struct_scaled = scaler.transform(struct)

    tfidf = vectorizer.transform([cleaned])

    final_features = hstack([tfidf, struct_scaled])

    score = model.predict(final_features)[0]

    return score, word_count, sentence_count, avg_sentence_length

# ================================
# FEEDBACK FUNCTION
# ================================
def get_feedback(score):
    if score < 2.5:
        return "Needs Improvement", "Improve clarity, grammar, and structure."
    elif score < 4:
        return "Average", "Good effort. Add stronger arguments."
    else:
        return "Strong", "Well-structured and clear essay."

# ================================
# UI
# ================================
st.title("📝 Automated Essay Scoring System")
st.write("Score Range: 1 (Low) to 6 (High)")

# Sidebar
st.sidebar.header("About")
st.sidebar.write("Model trained using ASAP dataset")
st.sidebar.write("Uses TF-IDF + structural features")

# Sample selection
sample = st.selectbox("Try sample essay", ["None"] + list(SAMPLE_ESSAYS.keys()))

if st.button("Load Sample"):
    if sample != "None":
        st.session_state["essay"] = SAMPLE_ESSAYS[sample]

# Input
essay = st.text_area("Enter your essay", height=250, key="essay")

# Live stats
if essay.strip():
    wc, sc, avg = get_stats(essay)
    c1, c2, c3 = st.columns(3)
    c1.metric("Words", wc)
    c2.metric("Sentences", sc)
    c3.metric("Avg Length", round(avg, 1))

# Prediction
if st.button("Predict Score"):

    if essay.strip() == "":
        st.warning("Please enter an essay.")
    else:
        with st.spinner("Analyzing essay..."):

            score, wc, sc, avg = predict(essay)

            # ================================
            # SCORE SCALING (IMPORTANT FIX)
            # ================================
            min_model_score = 1.0
            max_model_score = 3.0

            adjusted_score = 1 + (score - min_model_score) * (5 / (max_model_score - min_model_score))
            adjusted_score = max(1, min(6, adjusted_score))

            quality, feedback = get_feedback(adjusted_score)

        # Output
        st.write(f"Raw Score: {round(score,2)}")
        st.success(f"Final Score: {round(adjusted_score,2)}")
        st.info(f"Quality: {quality}")
        st.write("**Feedback:**", feedback)

        # Stats again
        c1, c2, c3 = st.columns(3)
        c1.metric("Words", wc)
        c2.metric("Sentences", sc)
        c3.metric("Avg Length", round(avg, 1))