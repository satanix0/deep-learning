"""Streamlit app for showing content-based movie recommendations."""

import streamlit as st

from omdb_utils import get_movie_details
from recommender import df, recommend_movies

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="centered",
)

st.title("🎬 Movie Recommender")

movie_list = sorted(df["title"].dropna().unique())
selected_movie = st.selectbox("Select a movie", options=movie_list)

if st.button("Recommend"):
    with st.spinner("Finding similar movies..."):
        recommendations = recommend_movies(selected_movie, top_n=5)
        if recommendations.empty:
            st.error("Sorry! No recommendations found.")
        else:
            st.success("Top recommendations:")
            for _, row in recommendations.iterrows():
                movie_title = row["title"]
                plot, poster = get_movie_details(movie_title)

                with st.container():
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if poster:
                            st.image(poster, width=100)
                        else:
                            st.write("No poster available")
                    with col2:
                        st.markdown(f"### {movie_title}")
                        st.markdown(f"*{plot}*" if plot else "_Plot not available_")
