import json, os
import numpy as np
import streamlit as st
from cleaning import clean_text

HERE = os.path.dirname(os.path.abspath(__file__))
cfg = json.load(open(os.path.join(HERE, "config.json")))

st.set_page_config(page_title="Logical Fallacy Detector", page_icon="🧠", layout="centered")

@st.cache_resource(show_spinner="Loading the model (first visit only)...")
def load_model():
    if cfg["use_transformer"]:
        from transformers import pipeline
        return pipeline("text-classification", model=cfg["model_repo"], device=-1)
    import joblib
    return joblib.load(os.path.join(HERE, "model.pkl"))

def scores(text):
    model = load_model()
    if cfg["use_transformer"]:
        res = model(text, top_k=None, truncation=True, max_length=cfg["max_len"])
        return {r["label"]: float(r["score"]) for r in res}
    x = [clean_text(text)]
    if hasattr(model, "predict_proba"):
        s = model.predict_proba(x)[0]
    else:                                   # Linear SVM: decision scores -> softmax
        d = model.decision_function(x)[0]
        s = np.exp(d - d.max()) / np.exp(d - d.max()).sum()
    return {str(c): float(v) for c, v in zip(model.classes_, s)}

EXAMPLES = [
    "My two cousins failed their driving tests, so people from our town are bad drivers.",
    "You can't trust his opinion on taxes, he can't even keep his garden tidy.",
    "Either you support this plan completely or you want the city to fail.",
    "Millions of people bought this diet book, it must work.",
]

st.title("🧠 Logical Fallacy Detector")
st.write("Type a short English argument. The model predicts which of 13 types of "
         "**logical fallacy** it contains and shows its 3 best guesses.")
st.caption(f"Model: {cfg['model_name']} (test macro-F1 = {cfg['test_macro_f1']:.3f}) · "
           "Data: LOGIC dataset (tasksource/logical-fallacy on Hugging Face)")

choice = st.selectbox("Try an example, or write your own below:", ["(write my own)"] + EXAMPLES)
text = st.text_area("Argument", value="" if choice == "(write my own)" else choice, height=120,
                    placeholder="e.g. Everyone is buying this phone, so it must be the best.")

if st.button("Detect fallacy", type="primary"):
    if not text.strip():
        st.warning("Please type an argument first.")
    else:
        top3 = sorted(scores(text).items(), key=lambda kv: kv[1], reverse=True)[:3]
        label, conf = top3[0]
        st.success(f"**Predicted fallacy: {label.title()}** ({conf:.0%} confidence)")
        st.write("Top 3 guesses:")
        for l, s in top3:
            st.progress(s, text=f"{l.title()} – {s:.0%}")

st.info("**Limits:** the model was trained only on texts that contain a fallacy, so it has no "
        "\"no fallacy\" answer and will always name one. The training texts are mostly short, "
        "textbook-style examples, so results on real-world posts may be worse. "
        "Treat the output as a suggestion.")
