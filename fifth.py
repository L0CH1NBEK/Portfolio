import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
credits = pd.read_csv('tmdb_5000_credits.csv')
movies = pd.read_csv('tmdb_5000_movies.csv')

credits_columns_renamed = credits.rename(index=str, columns= {"movie_id": "id"})
movie_merge = movies.merge(credits_columns_renamed, on = 'id')
#print(movie_merge.head())

movies_cleaned = movie_merge.drop(columns = ['homepage', 'title_x', 'title_y', 'status', 'production_countries'])


#tfv = TfidfVectorizer(mid_df=3, max_features = None,
#                     strip_accents = 'unicode', analyser = 'word', token_pattern = r'\w{1,}',
#                     ngram_range = (1,3),
#                     stop_words = 'english')
tfv = TfidfVectorizer(min_df = 3, max_features=None,  strip_accents = 'unicode',
                      analyzer = 'word',
                      token_pattern =r'\w{1,}',
                      ngram_range = (1,3),
                      stop_words = 'english')
#tfv_matrix = tfv.fit_transform(movies_cleaned['overview'])
tfv_matrix = tfv.fit_transform(movies_cleaned['overview'].values.astype('U'))

print(tfv_matrix)
print(tfv_matrix.shape)


sig = sigmoid_kernel(tfv_matrix, tfv_matrix)
#print(sig[0])

indices = pd.Series(movies_cleaned.index, index = movies_cleaned['original_title']).drop_duplicates()

print(list(enumerate(sig[indices['Newlyweds']])))

print(sorted(list(enumerate(sig[indices['Newlyweds']])), key=lambda x: x[1], reverse=True))

def give_recomendations(title, sig = sig):
    
    idx = indices[title]
    
    sig_score = list(enumerate(sig[idx]))
    
    sig_score = sorted(sig_score, key=lambda x:x[1], reverse = True)
    
    sig_score = sig_score[1:11]
    
    movie_indices = [i[0] for i in sig_score]
    
    return movies_cleaned['original_title'].iloc[movie_indices]
    
    
print(give_recomendations('Shrek'))





