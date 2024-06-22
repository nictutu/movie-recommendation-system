import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the datasets
movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')

# Merge datasets on movieId
data = pd.merge(ratings, movies, on='movieId')

# Create a user-item interaction matrix
user_movie_matrix = data.pivot_table(index='userId', columns='title', values='rating').fillna(0)

# Perform dimensionality reduction using TruncatedSVD
svd = TruncatedSVD(n_components=50, random_state=42)
reduced_matrix = svd.fit_transform(user_movie_matrix)

# Calculate the cosine similarity matrix
cosine_sim = cosine_similarity(reduced_matrix)

def recommend_movies(user_id, user_movie_matrix, cosine_sim, num_recommendations=5):
    # Get the index of the user's ratings
    user_index = user_id - 1  # Adjusting for zero-based indexing
    
    # Get the pairwise similarity scores of all users with the target user
    similarity_scores = list(enumerate(cosine_sim[user_index]))
    
    # Sort users based on similarity scores
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the most similar users
    similar_users = [i[0] for i in similarity_scores[1:num_recommendations+1]]
    
    # Get the movie ratings of the most similar users
    similar_users_ratings = user_movie_matrix.iloc[similar_users]
    
    # Calculate the average ratings for each movie
    avg_ratings = similar_users_ratings.mean(axis=0)
    
    # Sort the movies based on average ratings
    recommended_movies = avg_ratings.sort_values(ascending=False)
    
    # Filter out movies the user has already rated
    user_rated_movies = user_movie_matrix.iloc[user_index]
    recommended_movies = recommended_movies[user_rated_movies == 0]
    
    return recommended_movies.head(num_recommendations).index.tolist()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    user_id = int(data['user_id'])
    recommendations = recommend_movies(user_id, user_movie_matrix, cosine_sim)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
