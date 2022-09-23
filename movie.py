import pandas as pd
import numpy as np
import re
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
nltk.download('punkt')
nltk.download('stopwords')
pd.set_option('display.max_columns', None)
df = pd.read_csv("IMDB_Top250Engmovies2_OMDB_Detailed.csv")
df.head()
# print(len(df))
df['clean_plot'] = df['Plot'].str.lower()
df['clean_plot'] = df['clean_plot'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))
df['clean_plot'] = df['clean_plot'].apply(lambda x: re.sub('\s+', ' ', x))

df['clean_plot'] = df['clean_plot'].apply(lambda x: nltk.word_tokenize(x))
# print(df['clean_plot'])

stop_words = nltk.corpus.stopwords.words('english')
plot = []
for sentence in df['clean_plot']:
    temp = []
    for word in sentence:
        if word not in stop_words and len(word) >= 3:
            temp.append(word)
    plot.append(temp)
# print(plot)
df['clean_plot'] = plot
# print(df['clean_plot'])
# print(df.head())
df['Genre'] = df['Genre'].apply(lambda x: x.split(','))
df['Actors'] = df['Actors'].apply(lambda x: x.split(',')[:4])
df['Director'] = df['Director'].apply(lambda x: x.split(','))
def clean(sentence):
    temp = []
    for word in sentence:
        temp.append(word.lower().replace(' ', ''))
    return temp
df['Genre'] = [clean(x) for x in df['Genre']]
df['Actors'] = [clean(x) for x in df['Actors']]
df['Director'] = [clean(x) for x in df['Director']]
# print(df['Actors'][0])
columns = ['clean_plot', 'Genre', 'Actors', 'Director']
l = []
for i in range(len(df)):
    words = ''
    for col in columns:
        words += ' '.join(df[col][i]) + ' '
    l.append(words)
df['clean_input'] = l
df = df[['Title', 'clean_input']]

tfidf = TfidfVectorizer()
features = tfidf.fit_transform(df['clean_input'])
# print(features)

cosine_sim = cosine_similarity(features, features)
# print(cosine_sim)
# print(cosine_sim)
index = pd.Series(df['Title'])
def recommend_movies(title):
    movies = []
    idx = index[index == title].index[0]
    # print(idx)
    score = pd.Series(cosine_sim[idx]).sort_values(ascending=False)
    top10 = list(score.iloc[1:11].index)
    # print(top10)
    
    for i in top10:
        movies.append(df['Title'][i])
    return movies
# print(recommend_movies('Spider-Man: Homecoming'))