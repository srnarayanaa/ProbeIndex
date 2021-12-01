# Import the libraries
import os
import pandas as pd
import numpy as np
import json
import re
import math
from numpy import dot


import nltk
from numpy.linalg import norm
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')


#Import data file
allSongs = pd.read_csv("allSongs.csv")

def clean_data(dataframe):
    dataframe.drop_duplicates(subset='track_name', inplace=True)

    dataframe['track_name'] = dataframe['track_name'].str.lower()
    dataframe['track_artist_name'] = dataframe['track_artist_name'].str.lower()
    
    dataframe['track_name'] = dataframe['track_name'].str.replace('[^\w\s]',' ')
    dataframe['track_artist_name'] = dataframe['track_artist_name'].str.replace('[^\w\s]',' ')

    return dataframe


def getCosineSimilarity(l1, l2): 
    cosSim = dot(l1, l2)/(norm(l1)*norm(l2))
    return cosSim


def searchBySong(songName):
    
    results = pd.DataFrame(columns = ['track_id', 'track_name', 'track_artist_name', 'track_popularity', 'track_lyrics'])
    results['Similarity'] = 0
    for i in range(0,len(allSongs)):
        print(songName)
        cosSim = getCosineSimilarity(allSongs.loc[i,'track_name'], songName)  
        if(cosSim > 0.0):
            results = results.append(allSongs.loc[i])
            results['Similarity'][i] = cosSim
    print(results)
    return results


        
searchBySong('Say')