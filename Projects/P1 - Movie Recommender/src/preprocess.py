"""Build cleaned text features and similarity artifacts for the movie app."""

import logging
import re
from pathlib import Path

import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = BASE_DIR / "logs"
ARTIFACT_DIR = BASE_DIR / "artifacts"
MOVIES_PATH = BASE_DIR / "movies.csv"

LOG_DIR.mkdir(exist_ok=True)
ARTIFACT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s - %(levelname)s - %(message)s]",
    handlers=[
        logging.FileHandler(LOG_DIR / "preprocess.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logging.info("Starting the preprocessing script")

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

try:
    df = pd.read_csv(MOVIES_PATH)
    logging.info("Dataset loaded successfully, total rows: %d", len(df))
except Exception:
    logging.exception("Error loading dataset")
    raise

stop_words = set(stopwords.words("english"))


def preprocess_text(text):
    """Normalize movie metadata into lowercase tokens without stopwords."""
    text = re.sub(r"[^a-zA-Z\s]", "", text).lower()
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)


required_columns = ["genres", "keywords", "overview", "title"]
df = df[required_columns].dropna().reset_index(drop=True)
df["combined"] = df["genres"] + " : " + df["keywords"] + " : " + df["overview"]

data = df[["title", "combined"]].copy()

logging.info("Preprocessing the text data")
try:
    data["clean_text"] = data["combined"].apply(preprocess_text)
    logging.info("Text data preprocessed successfully")
except Exception:
    logging.exception("Error preprocessing text data")
    raise

try:
    logging.info("Vectorizing the text data using TF-IDF")
    tf_idf = TfidfVectorizer(max_features=5000, stop_words="english")
    tfidf_matrix = tf_idf.fit_transform(data["clean_text"])
    logging.info("Text data vectorized successfully")
except Exception:
    logging.exception("Error vectorizing text data")
    raise

try:
    logging.info("Calculating cosine similarity matrix")
    cosine_sim = cosine_similarity(tfidf_matrix)
    logging.info("Cosine similarity matrix calculated successfully")
except Exception:
    logging.exception("Error calculating cosine similarity matrix")
    raise

logging.info("Saving the TF-IDF vectorizer and cosine similarity matrix to disk")
joblib.dump(data[["title", "clean_text"]], ARTIFACT_DIR / "cleaned_data.pkl")
joblib.dump(tfidf_matrix, ARTIFACT_DIR / "tfidf_matrix.pkl")
joblib.dump(cosine_sim, ARTIFACT_DIR / "cosine_sim.pkl")

logging.info("Preprocessing script completed successfully")
