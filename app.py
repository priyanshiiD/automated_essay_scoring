import pickle
import re

import streamlit as st
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from scipy.sparse import hstack

import nltk

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

SAMPLE_ESSAYS = {
    "School Uniforms": (
        "School uniforms should be required in all public schools because they help students focus on learning "
        "instead of fashion. When everyone wears the same clothes, social pressure is reduced and students feel "
        "less judged for what they can or cannot afford. Uniforms can also improve discipline because they create "
        "a sense of belonging and responsibility. Some people argue that uniforms limit self-expression, but "
        "students can still express themselves through ideas, activities, and behavior. Overall, uniforms create "
        "a fairer and more focused environment for education."
    ),
    "Online Learning": (
        "Online learning has changed education by making classes more flexible and accessible. Students can attend "
        "lessons from home, review recordings, and learn at their own pace. This is especially helpful for students "
        "who live far from schools or have part-time responsibilities. However, online learning also has challenges. "
        "Many students struggle with distractions, weak internet, and less face-to-face interaction. Teachers must "
        "use engaging methods and clear communication to keep students motivated. In my view, the best approach is "
        "a blended model where online tools support classroom teaching."
    ),
    "Social Media Impact": (
        "Social media has both positive and negative effects on teenagers. On the positive side, it helps young "
        "people connect with friends, share ideas, and learn about global events. It can also provide communities "
        "for students with similar interests. On the negative side, excessive use may reduce concentration, disturb "
        "sleep, and increase anxiety due to constant comparison. Teenagers may also face misinformation and cyberbullying. "
        "Parents and schools should teach digital responsibility, while students should set healthy limits for screen time. "
        "Used wisely, social media can be a useful tool rather than a harmful distraction."
    ),
}

st.set_page_config(
    page_title="Automated Essay Scoring System",
    page_icon="📝",
)

# Load saved files
model = pickle.load(open("essay_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)


def get_essay_stats(text):
    essay_length = len(text.split())
    sentence_count = len(sent_tokenize(text)) if text.strip() else 0
    avg_sentence_length = essay_length / max(sentence_count, 1)
    char_count = len(text)
    return {
        "word_count": essay_length,
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_length,
        "char_count": char_count,
    }


def predict(text):
    cleaned = clean_text(text)
    stats = get_essay_stats(text)

    struct = [[stats["word_count"], stats["sentence_count"], stats["avg_sentence_length"]]]
    struct_scaled = scaler.transform(struct)
    tfidf = vectorizer.transform([cleaned])
    final_features = hstack([tfidf, struct_scaled])
    return model.predict(final_features)[0], stats


def get_quality_feedback(score):
    # Use broad score bands that work across common essay-score scales.
    if score <= 10:
        low_cutoff, medium_cutoff = 4, 7
    elif score <= 30:
        low_cutoff, medium_cutoff = 12, 20
    else:
        low_cutoff, medium_cutoff = 20, 35

    if score < low_cutoff:
        return "Low", "Try improving clarity, grammar, and overall structure."
    if score < medium_cutoff:
        return "Medium", "Good attempt. Add stronger argument flow and examples."
    return "High", "Strong writing quality. Keep polishing precision and coherence."


# Simple UI (kept close to original)
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1050px;
        }
        .simple-card {
            border: 1px solid #e6e9ef;
            border-radius: 12px;
            padding: 1rem;
            background: #ffffff;
        }
        .result-card {
            border: 1px solid #d6e4ff;
            border-radius: 12px;
            padding: 1rem;
            background: #f7faff;
            margin-top: 0.75rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Automated Essay Scoring System")
st.caption("Enter your essay and get an instant predicted score.")

if "essay_input" not in st.session_state:
    st.session_state["essay_input"] = ""

with st.sidebar:
    st.subheader("About")
    st.write("Model trained for automated essay scoring.")
    st.markdown("Dataset: [ASAP 2.0](https://www.kaggle.com/datasets/lburleigh/asap-2-0)")

st.markdown('<div class="simple-card">', unsafe_allow_html=True)
sample_choice = st.selectbox(
    "Try a sample essay",
    ["None"] + list(SAMPLE_ESSAYS.keys()),
    help="Pick a sample and click load to fill the essay box.",
)

if st.button("Load Sample Essay", key="load_sample_btn"):
    if sample_choice != "None":
        st.session_state["essay_input"] = SAMPLE_ESSAYS[sample_choice]
    else:
        st.info("Select a sample first.")

essay = st.text_area(
    "Enter your essay",
    height=240,
    placeholder="Write your essay here...",
    key="essay_input",
)

live_stats = get_essay_stats(essay) if essay.strip() else None
if live_stats:
    c1, c2, c3 = st.columns(3)
    c1.metric("Words", live_stats["word_count"])
    c2.metric("Sentences", live_stats["sentence_count"])
    c3.metric("Avg sentence length", f"{live_stats['avg_sentence_length']:.1f}")

predict_btn = st.button("Predict Score", use_container_width=True, key="predict_btn")
st.markdown("</div>", unsafe_allow_html=True)

if predict_btn:
    if essay.strip() == "":
        st.warning("Please enter an essay")
    else:
        score, stats = predict(essay)
        quality, feedback = get_quality_feedback(score)
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.success(f"Predicted Score: {round(score, 2)}")
        st.info(f"Quality: {quality} | {feedback}")
        r1, r2, r3 = st.columns(3)
        r1.metric("Words", stats["word_count"])
        r2.metric("Sentences", stats["sentence_count"])
        r3.metric("Avg sentence length", f"{stats['avg_sentence_length']:.1f}")
        st.markdown("</div>", unsafe_allow_html=True)