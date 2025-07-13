import requests
import json
import logging


config = json.load(open("api_config.json"))
OMDB_API_KEY = config["OMDB_API_KEY"]


def get_movie_details(movie_title):
    url = f"http://www.omdbapi.com/?apikey={OMDB_API_KEY}&t={movie_title}"
    response = requests.get(url).json()
    if response.get("Response") == "True":
        result = {"plot": response.get("Plot"),
                  "poster": response.get("Poster")}
        return result["plot"], result["poster"]

    return None, None


print(get_movie_details("The Matrix"))
