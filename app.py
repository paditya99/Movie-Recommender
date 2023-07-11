import streamlit as st
import pickle
import pandas as pd
import requests
from PIL import Image
import ast
import numpy as np
import pandas as pd

movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')

movies=movies.merge(credits, on='title')
movies= movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.dropna(inplace=True)


#function to convert string array object into the list
def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies['genres']= movies['genres'].apply(convert)
movies['keywords']= movies['keywords'].apply(convert)

#function to convert string array object into the list and fetch top 3 actors
def convert2(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if(counter!=3):
            L.append(i['name'])
            counter=counter+1
        else:
            break
    return L

movies['cast']= movies['cast'].apply(convert2)

#function to convert string array object into the list and fetch director from the crew
def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if(i['job']=='Director'):
            L.append(i['name'])
            break
    return L


movies['crew']= movies['crew'].apply(fetch_director)
#splitting overview string into a list
movies['overview']=movies['overview'].apply(lambda x: x.split())

#removing whitespaces between words
movies['genres']=movies['genres'].apply(lambda x: [i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x: [i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x: [i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x: [i.replace(" ","") for i in x])

movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']
new_df=movies[['movie_id','title','tags']]
new_df['tags']=new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags']=new_df['tags'].apply(lambda x: x.lower())

!pip install nltk
#stemming over tags
import nltk

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

def stemmer(text):
    L=[]
    for i in text.split():
        L.append(ps.stem(i))
    return " ".join(L)

new_df['tags']=new_df['tags'].apply(stemmer)

#converting each tags into the bag of words(word_vectorization)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,  stop_words='english')

vectors= cv.fit_transform(new_df['tags']).toarray()

from sklearn.metrics.pairwise import cosine_similarity
similarity= cosine_similarity(vectors)











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
