import streamlit as st
import pickle
import pandas as pd
import requests
from PIL import Image

pickle.dump(new_df,open('movies.pkl', 'wb'))
pickle.dump(new_df.to_dict(),open('movie_dict.pkl', 'wb'))
pickle.dump(similarity,open('similarity.pkl', 'wb'))

def fetch_posters(movie_id):
    response = requests.get(
        'https://api.themoviedb.org/3/movie/{}?api_key=7e3c1d993e7e6c1da2f5eb5dc7ef428f'.format(movie_id))
    data = response.json()
    return "https://image.tmdb.org/t/p/w500/" + data['poster_path']


def recommender(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = similarity[index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_movies_posters = []

    for i in movie_list:
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_movies_posters.append(fetch_posters(movies.iloc[i[0]].movie_id))
    return recommended_movies, recommended_movies_posters


movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)

similarity = pickle.load(open('similarity.pkl', 'rb'))

st.title('Movie Recommender System')
selected_name = st.selectbox('Select your movie', movies['title'].values)

if st.button('Recommend'):
    movie_title, posters = recommender(selected_name)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(movie_title[0])
        st.image(posters[0])
    with col2:
        st.text(movie_title[1])
        st.image(posters[1])
    with col3:
        st.text(movie_title[2])
        st.image(posters[2])
    with col4:
        st.text(movie_title[3])
        st.image(posters[3])
    with col5:
        st.text(movie_title[4])
        st.image(posters[4])
