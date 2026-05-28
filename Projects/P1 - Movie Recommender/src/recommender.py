"""Recommendation helpers backed by precomputed movie similarity artifacts."""

import logging
from pathlib import Path

import joblib

BASE_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = BASE_DIR / "logs"
ARTIFACT_DIR = BASE_DIR / "artifacts"

LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s - %(levelname)s - %(message)s]",
    handlers=[
        logging.FileHandler(LOG_DIR / "recommender.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

logging.info("Starting the recommender script")
logging.info("Loading the cleaned data and cosine similarity matrix")

try:
    df = joblib.load(ARTIFACT_DIR / "cleaned_data.pkl")
    cosine_sim = joblib.load(ARTIFACT_DIR / "cosine_sim.pkl")
    logging.info("Data loaded successfully, total rows: %d", len(df))
except Exception:
    logging.exception("Error loading data")
    raise


def recommend_movies(query, top_n=5):
    """Return the top N movies most similar to the selected movie title."""
    logging.info("Recommending movies for: %s", query)

    match_index = df[df["title"].str.lower() == query.lower()].index
    if len(match_index) == 0:
        return df[["title"]].iloc[0:0].copy()

    movie_index = match_index[0]
    similarity_scores = list(enumerate(cosine_sim[movie_index]))
    similarity_scores = sorted(similarity_scores, key=lambda score: score[1], reverse=True)
    similarity_scores = similarity_scores[1 : top_n + 1]
    logging.info("Top %d recommendations ready.", top_n)

    movie_indices = [index for index, _ in similarity_scores]
    recommended_results = df[["title"]].iloc[movie_indices].reset_index(drop=True)
    recommended_results.index = recommended_results.index + 1
    recommended_results.index.name = "Rank"
    return recommended_results
