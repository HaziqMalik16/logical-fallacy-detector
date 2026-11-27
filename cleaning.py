import json, os, re

_cfg = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")))
_STOP = {"all": set(_cfg["stop_words_all"]), "cue": set(_cfg["stop_words_cue"])}

def expand_contractions(text):
    text = text.replace("’", "'").replace("‘", "'")
    text = re.sub(r"\bcan't\b", "cannot", text, flags=re.I)
    text = re.sub(r"\bwon't\b", "will not", text, flags=re.I)
    text = re.sub(r"\bshan't\b", "shall not", text, flags=re.I)
    text = re.sub(r"n't\b", " not", text, flags=re.I)
    text = re.sub(r"'re\b", " are", text, flags=re.I)
    text = re.sub(r"'ve\b", " have", text, flags=re.I)
    text = re.sub(r"'ll\b", " will", text, flags=re.I)
    text = re.sub(r"'m\b", " am", text, flags=re.I)
    return text

def clean_text(text, stopword_mode=None):
    mode = stopword_mode or _cfg["stopword_mode"]
    text = expand_contractions(str(text))
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = text.lower()
    if mode in _STOP:
        text = " ".join(w for w in text.split() if w not in _STOP[mode])
    return " ".join(text.split())
