# preprocess.py
import pandas as pd
import re
import nltk
import joblib
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s - %(levelname)s - %(message)s]',
                    handlers=[
                        logging.FileHandler(
                            '..//logs//omdb_api.log', encoding="utf-8"),
                        logging.StreamHandler()
                    ]
                    )
logging.info("Starting the preprocessing script")

nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
try:
    df = pd.read_csv('..//movies.csv')
    logging.info("Dataset loaded successfully, total rows: %d", len(df))
except Exception as e:
    logging.error("Error loading dataset: %s", str(e))
    raise e

stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    # Remove special characters and digits, them convert to lowercase
    text = re.sub(r"[^a-zA-Z\s]", "", text).lower()
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    token = [word for word in tokens if word not in stop_words]
    return " ".join(token)


# Filter the dataset to keep only the relevant columns
# and drop rows with missing values
required_columns = ['genres', 'keywords', 'overview', 'title']
df = df[required_columns]
df = df.dropna().reset_index(drop=True)
# Combine the text data into a single column
df['combined'] = df['genres'] + ' : ' + df['keywords'] + ' : ' + df['overview']

data = df[['title', 'combined']]

# Clean the text data
logging.info("Preprocessing the text data")
try:
    # Preprocess the text data
    data['clean_text'] = data['combined'].apply(preprocess_text)
    logging.info("Text data preprocessed successfully")
except Exception as e:
    logging.error("Error preprocessing text data: %s", str(e))
    raise e

# Vectorize the text data using TF-IDF
try:
    logging.info("Vectorizing the text data using TF-IDF")
    tf_idf = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = tf_idf.fit_transform(data['clean_text'])
    logging.info("Text data vectorized successfully")
except Exception as e:
    logging.error("Error vectorizing text data: %s", str(e))
    raise e

# Cosine similarity matrix
try:
    logging.info("Calculating cosine similarity matrix")
    cosine_sim = cosine_similarity(tfidf_matrix)
    logging.info("Cosine similarity matrix calculated successfully")
except Exception as e:
    logging.error("Error calculating cosine similarity matrix: %s", str(e))
    raise e

# Save the TF-IDF vectorizer and cosine similarity matrix to disk
logging.info(
    "Saving the TF-IDF vectorizer and cosine similarity matrix to disk")
joblib.dump(data[['title', 'clean_text']], '..//artifacts//cleaned_data.pkl')
joblib.dump(tfidf_matrix, '..//artifacts//tfidf_matrix.pkl')
joblib.dump(cosine_sim, '..//artifacts//cosine_sim.pkl')


logging.info("Preprocessing script completed successfully")
