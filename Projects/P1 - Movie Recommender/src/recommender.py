import logging
import joblib

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s - %(levelname)s - %(message)s]',
                    handlers=[
                        logging.FileHandler(
                            '..//logs//omdb_api.log', encoding="utf-8"),
                        logging.StreamHandler()
                    ]
                    )

logging.info("Starting the recommender script")
logging.info("Loading the cleaned data and cosine similarity matrix")

try:
    df = joblib.load('..//artifacts//cleaned_data.pkl')
    cosine_sim = joblib.load('..//artifacts//cosine_sim.pkl')
    logging.info("Data loaded successfully, total rows: %d", len(df))
except Exception as e:
    logging.error("Error loading data: %s", str(e))
    raise e


# recommendation function
def recommend_movies(query, top_n=5):
    logging.info(f"🎬 Recommending movies for: {query}", )

    # Get the index of the movie from the query
    idx = df[df['title'].str.lower() == query.lower()].index
    if len(idx) == 0:
        return "Movie not found in the database."
    idx = idx[0]
    # get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    # sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # get the scores of the top n most similar movies
    sim_scores = sim_scores[1:top_n+1]
    logging.info("✅ Top %d recommendations ready.", top_n)

    movie_indices = [i[0] for i in sim_scores]
    recommended_results = df[['title']
                             ].iloc[movie_indices].reset_index(drop=True)
    recommended_results.index = recommended_results.index + 1
    recommended_results.index.name = 'Rank'
    return recommended_results
