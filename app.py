import json, os
import numpy as np
import streamlit as st
from cleaning import clean_text

HERE = os.path.dirname(os.path.abspath(__file__))
cfg = json.load(open(os.path.join(HERE, "config.json")))

st.set_page_config(page_title="Logical Fallacy Detector", page_icon="🧠", layout="centered")

# The 13 fallacy classes of the LOGIC dataset: other common name, meaning, example
FALLACIES = {
    "ad hominem": ("Personal attack",
        "Attacks the person making the argument instead of the argument itself.",
        "You can't trust his opinion on taxes, he can't even keep his garden tidy."),
    "ad populum": ("Bandwagon / appeal to popularity",
        "Claims something is true or good because many people believe it or do it.",
        "Millions of people bought this diet book, so it must work."),
    "appeal to emotion": ("Appeal to fear, pity or pride",
        "Tries to win with feelings such as fear, pity, guilt or pride instead of reasons.",
        "If you don't buy this alarm system, your family could be in danger tonight."),
    "circular reasoning": ("Begging the question",
        "The conclusion is already assumed in the reason given for it.",
        "This book is true because the book itself says it never lies."),
    "equivocation": ("Double meaning",
        "Uses the same word in two different meanings to make a weak argument look valid.",
        "A feather is light. What is light cannot be dark. So a feather cannot be dark."),
    "fallacy of credibility": ("Appeal to authority",
        "Accepts a claim because of who says it (a celebrity, an unrelated expert), not because of evidence.",
        "A famous actor says this toothpaste is the best, so it must be true."),
    "fallacy of extension": ("Straw man",
        "Twists or exaggerates the other side's argument, then attacks that weaker version.",
        "She wants fewer sweets in the canteen, so she wants to ban all fun at school."),
    "fallacy of logic": ("Deductive / formal fallacy",
        "The conclusion does not follow from the reasons, because the logical step itself is wrong.",
        "If it rains, the road is wet. The road is wet, so it must have rained."),
    "fallacy of relevance": ("Red herring",
        "Brings in an unrelated point to distract from the real issue.",
        "Why worry about my homework when there are so many problems in the world?"),
    "false causality": ("False cause / post hoc",
        "Assumes that because one thing came before or with another, it caused it.",
        "I wore my blue shirt and we won, so the shirt brings us luck."),
    "false dilemma": ("Either-or / black-and-white",
        "Presents only two options when more are possible.",
        "Either you support this plan completely or you want the city to fail."),
    "faulty generalization": ("Hasty generalization",
        "Draws a broad conclusion from too few or untypical examples.",
        "My two cousins failed their driving tests, so people from our town are bad drivers."),
    "intentional": ("Deliberately misleading argument",
        "Uses a trick on purpose to support a claim, e.g. treating a lack of proof as proof, or a loaded question.",
        "No one has proven that ghosts don't exist, so they must be real."),
}

def info(label):
    return FALLACIES.get(label.lower(), ("", "No description available.", ""))

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
        other_name, meaning, example = info(label)
        st.success(f"**Predicted fallacy: {label.title()}** ({other_name}) – {conf:.0%} confidence")
        st.markdown(f"**What it means:** {meaning}  \n**Typical example:** *\"{example}\"*")
        st.markdown("**Does it fit?** Compare your text with the meaning above. "
                    "If it does not fit, check the other two guesses:")
        for l, s in top3:
            st.progress(s, text=f"{l.title()} – {s:.0%}")
        for l, _ in top3[1:]:
            n, m, _ = info(l)
            st.markdown(f"- **{l.title()}** ({n}): {m}")

st.info("**Limits:** the model was trained only on texts that contain a fallacy, so it has no "
        "\"no fallacy\" answer and will always name one. The training texts are mostly short, "
        "textbook-style examples, so results on real-world posts may be worse. "
        "Treat the output as a suggestion.")

st.subheader("📖 The 13 logical fallacies")
st.write("The model chooses one of these classes. The names in the first column are the labels used by "
         "the dataset; the second column gives the more common name.")
rows = ["| Fallacy (label) | Also called | What it means | Example |", "|---|---|---|---|"]
for name, (other, meaning, example) in FALLACIES.items():
    rows.append(f"| **{name.title()}** | {other} | {meaning} | *\"{example}\"* |")
st.markdown("\n".join(rows))
