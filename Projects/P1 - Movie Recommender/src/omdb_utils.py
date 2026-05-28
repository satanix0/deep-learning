"""Utilities for fetching movie metadata from the OMDb API."""

import json
import os
from pathlib import Path

import requests

CONFIG_PATH = Path(__file__).resolve().with_name("api_config.json")
OMDB_API_KEY = os.getenv("OMDB_API_KEY")

if not OMDB_API_KEY and CONFIG_PATH.exists():
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        OMDB_API_KEY = json.load(config_file).get("OMDB_API_KEY")


def get_movie_details(movie_title):
    """Return the plot and poster URL for a movie title, if OMDb can find it."""
    if not OMDB_API_KEY:
        return None, None

    response = requests.get(
        "https://www.omdbapi.com/",
        params={"apikey": OMDB_API_KEY, "t": movie_title},
        timeout=10,
    ).json()

    if response.get("Response") == "True":
        poster = response.get("Poster")
        if poster == "N/A":
            poster = None
        return response.get("Plot"), poster

    return None, None
