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
    "Venus Essay 1": (
        "The author suggests that studying Venus is worthy enough even though it is very dangerous. "
        "The author mentioned that on the planet's surface, temperatures average over 800 degrees Fahrenheit, "
        "and the atmospheric pressure is 90 times greater than what we experience on our own planet. "
        "His solution to survive this weather that is dangerous to us humans is to allow them to float above "
        "the fray. A \"blimp-like\" vehicle hovering 30 or so miles would help avoid the unfriendly ground "
        "conditions. At thirty-plus miles above the surface, temperatures would still be toasty at around "
        "170 degrees Fahrenheit, but the air pressure would be close to that of sea level on Earth. So not "
        "easy conditions, but survivable enough for humans. So this would help make the mission capeable "
        "of completing.\n\n"
        "He also mentions how peering at venus from a ship orbiting or hovering safely far above the planet "
        "can provide only limited insight on ground conditions because most forms of light cannot penertrate "
        "the dense atmosphere making it hard to take photographs. They also cannot take samples of rock, gas, "
        "or anything else, from a distance. So many reaserchers are working on innovations that would allow "
        "their machines to last long enough to help gain some imformation of Venus.\n\n"
        "They are working on other ways to study Venus such as simplified electrnics made of silicon carbide "
        "that have been tested in a chamber simulating the chaos of Venus's surface. So far they have lasted "
        "for 3 weeks in these conditions which is more than enough time hopefully for them to be able to grab "
        "enough information. Their other project that they are working on is using an old technology called "
        "mechanical computers. They are powerful, flexible, and quick. Systems that use mechanical parts can "
        "be made more resistant to pressure, heat, and other forces.\n\n"
        "He feels that studying Venus even though its dangerous is valuable because of the insight they could "
        "gain about the planet itself but also becuase \"human curiosity will likely lead us into many equally "
        "intimidating endeavors.\"\n\n"
        "I think the author supported his claim very well he explained why he thought it as nessary to go even "
        "though it is dangerous and he gave solutions to some of the dangers on Venus such as sollution to the "
        "heat and ways to actually help gain evicence and imformation on Venus."
    ),
    "Venus Essay 2": (
        "The author's claim of studying Venus is a worthy pursuit because Venus is closely related to Earth, "
        "Venus has a enviroment that is similar to Earth, and scientists want to explore more of what Venus "
        "has to offer.\n\n"
        "The first claim of why the author supports scientists studying Venus is that Venus is closely related "
        "to Earth. In the passage, it states, \"Often referred to as Earth's twin, Venus is the closest "
        "planet to Earth in terms of density and size, and occasionally the closest planet to Earth in terms "
        "of density and size, and occasionally the closest in distance too. Earth, Venus, and Mars our other "
        "planetary neighbors orbit the sun at different speeds. These differences in speed mean that sometimes "
        "we are closer to Mars and other times to Venus. Because Venus is sometimes right around the corner, "
        "in space terms, humans have sent numerous to land on this cloud draped world. Each precious mission "
        "was unmanned, and for good reason, since no spacecraft survived the landing for more than a few hours. "
        "Maybe this issue explains why not a single spaceship has touched down on Venus in more than three "
        "decades. Numerous factors contribute to Venus's reputation as a challenging planet for humans to study, "
        "despite its proximity to us\" (Paragraph 2). This supports the author's claim because the author "
        "believes that scientists should study Venus because Venus is closely related to Earth.\n\n"
        "The second claim of why the author supports scientists studying Venus is that Venus has a similar "
        "enviroment to Earth. In the passage, it states, \"A thick atmosphere of almost 97 percent carbon "
        "dioxide blankets Venus. Even more challenging are the clouds of highly corrosive sulfuric acid in "
        "Venus's atmosphere. On the planet's surface, temperatures average over 800 degrees Fahrenheit, and "
        "the atmospheric pressure is 90 times greater than what we experience on our own planet. These conditions "
        "are far more extreme than anything humans encounter on Earth; such an enviroment would crush even a "
        "submarine accustomed to diving to the deepest parts of our oceans and would liquefy many metals. Also "
        "notable, Venus has the hottest surface temperature of any planet in our solar system, even though "
        "Mercury is closer to our sun. Beyond high pressure and heat, Venusian geology and weather present "
        "additional impediments like erupting volcanoes, powerful earthquakes, and frequent lightning strikes "
        "to probes seeking to land on its surface\" (Paragraph 3). This supports the author's claim because "
        "there are many physical and enviromental dangers that are present on Venus. Despite all of Venus's "
        "dangers, scientists still want to explore Venus in depth.\n\n"
        "The third claim of why the author supports scientists studying Venus is that the author supports the "
        "further exploration of Venus by scientists. In the passage it states, \"If our sister planet is so "
        "inhospitable, why are scientists even discussing futher visits to its surface? Astronomers are fascinated "
        "by Venus because it may well once have been the most Earth-like planet in our solar system. Long ago, "
        "Venus was probably covered largely with oceans and could have supported various forms of life, just "
        "like Earth. Today, Venus still has some features that are analogous to those on Earth. The planet has "
        "a surface of rocky sediment and includes familiar features such as valleys, mountains, and craters. "
        "Furthermore, recall that Venus can sometimes be our nearest option to visit, a crucial consideration "
        "given the long time frames of space travel. The value of returning to Venus seems indisputable, but "
        "what are the options for making such a mission both safe and scientifically productive?. NASA has one "
        "particulary conpelling idea for sending humans to study Venus. NASA's possible solution to the hostile "
        "conditions on the surface of Venus would allow scientists to float above the fray. Imagine a blimp-like "
        "vechile hovering over Venus would aviod the unfrendily ground conditions by staying up and out of thier "
        "way. At thirty-plus miles above the surface, temperature would still be toasty at around 170 degrees "
        "Fahrenheit, but the air pressure would be close to that of sea level on Earth. Solar power would be "
        "plentiful, and radiation would not exceed Earth levels. Not easy, conditions, but survivable for humans.\" "
        "(Paragraph 4 and 5). This supports the author's claim because scientists want to know more about Venus's "
        "planatery history and similarities to Earth.This statement also supports the author's claim because the "
        "scientists wants to try and send humans to Venus.\n\n"
        "Therefore, the author's claim supports the idea that studying Venus is a worthy pursuit despite the dangers "
        "it presents by Venus being closely related to Earth, Venus has a similar enviorment to Earth, and the author "
        "encouriging further explorations to Venus."
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