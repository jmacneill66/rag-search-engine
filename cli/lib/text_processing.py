# lib/text_processing.py

import string
from pathlib import Path
from nltk.stem import PorterStemmer

PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)

STOPWORDS_PATH = Path("data/stopwords.txt")
with STOPWORDS_PATH.open() as f:
    STOP_WORDS = set(f.read().splitlines())

stemmer = PorterStemmer()


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(PUNCTUATION_TABLE)
    return text


def tokenize_and_stem(text: str) -> list[str]:
    tokens = preprocess_text(text).split()
    return [
        stemmer.stem(token)
        for token in tokens
        if token and token not in STOP_WORDS
    ]
