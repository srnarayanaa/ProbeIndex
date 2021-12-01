import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize, RegexpTokenizer
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
english = WordNetLemmatizer()

def function(query):

    df = pd.read_csv('allSongs.csv')
    df = df.drop('track_lyrics', 1)
    song_dict = {}
    for i in range(len(df)):
        id = df['track_id'][i]
        auth = df['track_artist_name'][i]
        title = df['track_name'][i]
        rem = set([",", "(", ")", "'", "!", "-"])
        title = str(title)
        a = ""
        for x in title:
            if x not in rem: a += x
        doc = a +  " " + auth 
        song_dict[id] = doc

    def_set = set()
    all_content = list()
    for id in song_dict:
        def_id = id 
        contents = song_dict[def_id]
        all_content.append(contents)
        temp = list(contents.split())
        def_set.update(temp)
  
    stop = set(stopwords.words('english'))
    temp = set(); terms = set()

    for entry in def_set:
        if entry.lower() not in stop:
            temp.add(entry)

    terms = set()
    for i in temp:
        terms.add(english.lemmatize(i.lower()))

    pipe = Pipeline([('count', CountVectorizer(vocabulary=terms)),('tfid', TfidfTransformer())]).fit(all_content) 
    tf = pipe['count'].transform(all_content).toarray()
    w = pipe['tfid'].fit_transform(tf).toarray()


    qline = []
    qdef = []
    q_sh = []
    for x in query:
        v = x
        q_sh.append(v)
        qdefset = set()
        temp = list(v.split())
        qdef.append(temp)

    qterms = []
    for qdefset in qdef:
        qt = set()
        for entry in qdefset:
            if entry.lower() not in stop:
                qt.add(entry)
        qterms.append(qt)



    qt_1 = CountVectorizer(vocabulary=terms, binary=False).transform([q_sh[0].lstrip().lower()]).toarray()
    q_idf_1 = TfidfTransformer().fit_transform(qt_1).toarray()

    n = np.matmul(w, np.transpose(q_idf_1)).reshape(1,-1) 
    sq = np.repeat(np.sum(np.square(q_idf_1)), w.shape[0])
    d = np.multiply(np.sum(np.square(w),axis=1), sq)

    cosine_sims = n/(d); cosine_sims = cosine_sims.ravel()
    np_df = df.to_numpy()
    if cosine_sims > 0.0:
        order = sorted(range(len(cosine_sims)), key=lambda i: cosine_sims[i], reverse=True)[:10]

    ans = []
    for i in order: 
        ans.append((np_df[i]))
    return ans
