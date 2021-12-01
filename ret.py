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


df = pd.read_csv('papers.csv')
df = df.head(1000)

index = {}
author_dict = {}
title_dict = {}
paper_dict = {}
abs_dict = {}
for i in range(len(df)):
  id = str(df['id'][i])
  auth = df['authors'][i]
  a = ""
  rem = set([",", "[", "]", "'"])
  auth = str(auth)
  for x in auth:
    if x not in rem:
      a += x
  doc = df['title'][i] +  " " + a + " " + df['categories'][i] + " " + df['abstract'][i]
  paper_dict[id] = doc
  index[i] = id
  author_dict[id] = auth
  title_dict[id] = df['title'][i]
  abs_dict[id] = df['abstract'][i]
  
def_set = set()
all_content = list()
for id in paper_dict:
  def_id = id 
  contents = paper_dict[def_id]
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

def search(query):
    query = [query]

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


    #Query 1 
    qt_1 = CountVectorizer(vocabulary=terms, binary=False).transform([q_sh[0].lstrip().lower()]).toarray()
    q_idf_1 = TfidfTransformer().fit_transform(qt_1).toarray()

    #calc cosine sim
    n = np.matmul(w, np.transpose(q_idf_1)).reshape(1,-1) 
    sq = np.repeat(np.sum(np.square(q_idf_1)), w.shape[0])
    d = np.multiply(np.sum(np.square(w),axis=1), sq)

    cosine_sims = n/(d); cosine_sims = cosine_sims.ravel()

    ans = []
    if max(cosine_sims) < 0.85:
        order = sorted(range(len(cosine_sims)), key=lambda i: cosine_sims[i], reverse=True)[:10]
        print("Ranks - Document Relation")
        for i in order: 
            temp = []
            id = index[i]
            tid = id
            if len(id) == 8:
                tid = '0' + tid
            temp.append('https://arxiv.org/abs/'+tid)
            temp.append(title_dict[id].title())
            temp.append(author_dict[id].title())
            temp.append(abs_dict[id].title())
            ans.append(temp)
    return ans

print(search('bidirectional gaussian relay network'))