import pandas as pd
import streamlit as st
import re
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import plotly.express as px
from hdfs import InsecureClient


# HDFS configuration
HDFS_URL = 'http://localhost:9870'  # Change this to your HDFS NameNode URL
client = InsecureClient(HDFS_URL, user='hadoop', timeout=30)
    
# Load movies data
@st.cache_data
def load_movies():
    # Create an empty list to hold the chunks
    chunks = []
    # Read the CSV in chunks and append each chunk to the list
    with client.read('/movies.csv', encoding='utf-8') as reader:
        for chunk in pd.read_csv(reader, chunksize=100000):
            chunks.append(chunk)
    # Concatenate all the chunks into a single DataFrame and return it
    return pd.concat(chunks, ignore_index=True)

# Load ratings data in chunks
# Load ratings data from HDFS in chunks

@st.cache_data
def load_ratings():
     # Create an empty list to hold the chunks
    chunks = []
    with client.read('/ratings.csv', encoding='utf-8') as reader:
        for chunk in pd.read_csv(reader, chunksize=100000):
            chunks.append(chunk)
    # Concatenate all the chunks into a single DataFrame and return it
    return pd.concat(chunks, ignore_index=True)

movies = load_movies()
ratings = load_ratings()

# Clean title function
def clean_title(title):
    return re.sub("[^a-zA-Z0-9 ]", "", title)

# Prepare HashingVectorizer
movies["clean_title"] = movies["title"].apply(clean_title)
vectorizer = HashingVectorizer(n_features=2**10, ngram_range=(1,2))
tfidf = vectorizer.fit_transform(movies["clean_title"])

# Search function
def search(title):
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = movies.iloc[indices].iloc[::-1]
    return results

# Find similar movies function
@st.cache_data
def find_similar_movies(movie_id):
    similar_users = set()
    similar_user_recs = pd.Series(dtype='int64')
    all_user_recs = pd.Series(dtype='int64')
    total_users = 0

    # Using the full ratings DataFrame instead of chunks
    ratings_iterator = ratings  # ratings is already a DataFrame

    # Find users who liked the specified movie
    similar_users = ratings_iterator[(ratings_iterator["movieId"] == movie_id) & (ratings_iterator["rating"] > 4)]["userId"].unique()
    
    # Find recommendations for similar users
    similar_user_recs = ratings_iterator[(ratings_iterator["userId"].isin(similar_users)) & (ratings_iterator["rating"] > 4)]["movieId"].value_counts()

    # Find all user recommendations
    all_user_recs = ratings_iterator[ratings_iterator["rating"] > 4]["movieId"].value_counts()

    # Calculate scores
    similar_user_recs = similar_user_recs / len(similar_users) if len(similar_users) > 0 else similar_user_recs
    similar_user_recs = similar_user_recs[similar_user_recs > 0.1]
    
    total_users = ratings_iterator["userId"].nunique()  # total unique users
    
    all_user_recs = all_user_recs / total_users if total_users > 0 else all_user_recs
    
    rec_percentages = pd.DataFrame({"similar": similar_user_recs, "all": all_user_recs}).fillna(0)
    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
    rec_percentages = rec_percentages.sort_values("score", ascending=False)
    
    return rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")[["score", "title", "genres"]]


# Function to get dataset insights
# Function to get dataset insights
@st.cache_data
def get_dataset_insights():
    genre_counts = movies["genres"].str.split("|", expand=True).stack().value_counts()
    top_genres = genre_counts.head(10)
    
    year_counts = movies["title"].str.extract(r"\((\d{4})\)")[0].value_counts().sort_index()
    
    # Calculate ratings distribution directly from the entire DataFrame
    ratings_dist = ratings["rating"].value_counts().sort_index()  # No need for chunking here
    
    return top_genres, year_counts, ratings_dist


# Streamlit app
st.title("\U0001F3A5" + "Movie Recommender")

# User input
movie_name = st.text_input("Enter a movie title:")

if movie_name:
    # Search for the movie
    results = search(movie_name)
    
    if not results.empty:
        # Display search results and recommendations in a single column
        col1, col2 = st.columns([2, 1])  # Adjust the ratio as needed
        
        with col1:
            st.subheader("Search Results:")
            st.dataframe(results[["title", "genres"]], width=None)  # Full width
        
            # Get recommendations
            movie_id = results.iloc[0]["movieId"]
            recommendations = find_similar_movies(movie_id)
            
            st.subheader("Recommended Movies:")
            st.dataframe(recommendations, width=None)  # Full width
        
        # Recommendation Insights
        st.subheader("Recommendation Insights")
        col1, col2 = st.columns(2)
        
        with col1:
            # Genre distribution
            genre_counts = recommendations["genres"].str.split("|", expand=True).stack().value_counts()
            fig_genres = px.pie(values=genre_counts.values, names=genre_counts.index, title="Genre Distribution of Recommended Movies")
            st.plotly_chart(fig_genres, use_container_width=True)
        
        with col2:
            # Score distribution
            fig_scores = px.histogram(recommendations, x="score", nbins=20, title="Distribution of Recommendation Scores")
            st.plotly_chart(fig_scores, use_container_width=True)
        
        # Top 5 recommendations
        top_5 = recommendations.head()
        fig_top5 = px.bar(top_5, x="title", y="score", title="Top 5 Recommended Movies")
        st.plotly_chart(fig_top5, use_container_width=True)
    else:
        st.warning("No movies found. Please try a different title.")
else:
    st.info("Enter a movie title to get recommendations and insights.")

# Dataset Insights
st.header("Dataset Insights")
top_genres, year_counts, ratings_dist = get_dataset_insights()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Top 10 Genres in the Dataset")
    fig_top_genres = px.bar(x=top_genres.index, y=top_genres.values, title="Top 10 Genres")
    st.plotly_chart(fig_top_genres, use_container_width=True)

with col2:
    st.subheader("Movies by Year")
    fig_years = px.line(x=year_counts.index, y=year_counts.values, title="Number of Movies by Year")
    st.plotly_chart(fig_years, use_container_width=True)

# New insightful graph
st.subheader("Rating Distribution")
fig_ratings = px.bar(x=ratings_dist.index, y=ratings_dist.values, title="Distribution of Ratings")
fig_ratings.update_xaxes(title="Rating")
fig_ratings.update_yaxes(title="Count")
st.plotly_chart(fig_ratings, use_container_width=True)