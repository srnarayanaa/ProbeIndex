# Import the libraries
import os
from nltk.corpus.reader.mte import xpath
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


allSongs = pd.read_csv("allSongs.csv")
#allSongs = clean_data(allSongs)

def getCosineSimilarity(text1, text2):
    
    words1 = word_tokenize(text1)
    words2 = word_tokenize(text2)
    sw = stopwords.words('english')
    ps = PorterStemmer()
    
    for i in range(0,len(words1)):
        words1[i] = ps.stem(words1[i])
        
    for i in range(0,len(words2)):
        words2[i] = ps.stem(words2[i])
        
    # remove stop words from the string
    words1_set = {w for w in words1 if not w in sw} 
    words2_set = {w for w in words2 if not w in sw}
    l1 =[]
    l2 =[]
    
    # form a set containing keywords of both strings 
    rvector = words1_set.union(words2_set) 
    for w in rvector:
        if w in words1_set: l1.append(1) # create a vector
        else: l1.append(0)
        if w in words2_set: l2.append(1)
        else: l2.append(0)

    
    cosSim = dot(l1, l2)/(norm(l1)*norm(l2))
    return cosSim



def searchByArtist(artistName):
    results = pd.DataFrame(columns = ['track_id', 'track_name', 'track_artist_name', 'track_popularity', 'track_lyrics'])
    
    for i in range(0,len(allSongs)):
        cosSim = getCosineSimilarity(allSongs.loc[i,'track_artist_name'], artistName)
                                     
        if(cosSim > 0.0):
            results = results.append(allSongs.loc[i])
            results['Similarity'] = cosSim
    id = []
    ans = []
    name = []
    art = []
    for x in results['track_id']:
        id.append(x)
    for x in results['track_artist_name']:
        art.append(x)
    for x in results['track_name']:
        name.append(x)

    for i in range(len(art)):
        temp = []
        temp.append('https://open.spotify.com/track/'+id[i])
        temp.append(art[i])
        temp.append(name[i])
        ans.append(temp)
        
    return ans[::-1]

def searchBySong(songName):
    results = pd.DataFrame(columns = ['track_id', 'track_name', 'track_artist_name', 'track_popularity', 'track_lyrics'])
    results['Similarity'] = 0
    for i in range(0,len(allSongs)):
        cosSim = getCosineSimilarity(allSongs.loc[i,'track_name'], songName)  
        if(cosSim > 0.0):
            results = results.append(allSongs.loc[i])
            results['Similarity'][i] = cosSim
    id = []
    ans = []
    name = []
    art = []
    
    for x in results['track_id']:
        id.append(x)
    for x in results['track_artist_name']:
        art.append(x)
    for x in results['track_name']:
        name.append(x)

    for i in range(len(art)):
        temp = []
        temp.append('https://open.spotify.com/track/'+id[i])
        temp.append(art[i])
        temp.append(name[i])
        ans.append(temp)
        
    return ans

print(searchByArtist('Harris'))