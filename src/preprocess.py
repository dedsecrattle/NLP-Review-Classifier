import spacy
import re

nlp = spacy.load("en_core_web_sm")


def clean_text(text: str) -> str:
    """Basic preprocessing: lowercase, remove URLs, lemmatize, remove stopwords"""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "URL", text)
    doc = nlp(text)
    return " ".join([t.lemma_ for t in doc if not t.is_stop and t.is_alpha])
