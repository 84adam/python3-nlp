import sys
import os
import gensim
import pandas as pd  
import numpy as np
from gensim import corpora, models
from scipy.spatial import distance

"""
SEARCH_BUILD: 
Using pre-trained model, build a corpus of candidate documents to return as search results to users using `search_app.py`.
Requires 'tokens' column (preprocessed text) in order to infer topics of candidate docs.
"""

def infer_topic(tokens):
    dict_new = dictionary.doc2bow(tokens)
    vector = model[dict_new]
    return vector

def interp_topics(vector):
    present = []
    for i in vector:
        t = i[0]
        present.append(t)
    all_t = [x for x in range(num_topics)]
    missing = [x for x in all_t if x not in present]

    if len(missing) > 0:
        for i in missing:
            missing_i = (i, 0.0)
            vector.append(missing_i)

    fixed = sorted(vector)
    return fixed

def jsdist(p, q):
    return distance.jensenshannon(p, q, base=None)

def all_jsd(vector, tp):
    aj = []
    for i in tp:
    	j = jsdist(vector, i)
    	aj.append(j[1])
    return aj

def pickle_df(df, pname):
    df.to_pickle(pname)

def unpickle_df(pname, df):
    new_df = pd.read_pickle(pname)
    return new_df

def load_model():
    filepath = os.getcwd()  
    filename_model = filepath + '/' + 'tf-lda.model'
    filename_dict = filepath + '/' + 'tf-lda.dict'
    model = gensim.models.LdaModel.load(filename_model)
    dictionary = corpora.Dictionary.load(filename_dict)
    return model, dictionary

if __name__ == '__main__':
  # load trained model and dictionary
  print("Loading model...")
  num_topics = int(input("How many topics were used to train this model? "))
  topic_model = load_model()
  model = topic_model[0]
  dictionary = topic_model[1]

  # BUILD CANDIDATE DOCS CORPUS USING PRE-TRAINED MODEL:
  corpus_filename = input("Enter name of corpus on which to model topic vectors: ")
  save_name = input("Save candidate corpus as (enter a filename[.pkl]): ")
  save_name = save_name + ".pkl"

  df = pd.read_pickle(corpus_filename)
  df = df.drop_duplicates(subset=['title', 'raw'], keep='first')

  tokens = df['tokens']
  topics = []
  for i in tokens:
    result = interp_topics(infer_topic(i))
    topics.append(result)
  title = df['title']
  raw = df['raw']
  df = pd.DataFrame({'title': [x for x in title], 'raw': [x for x in raw], 'tokens': [x for x in tokens], 'topics': [x for x in topics]})
  df.to_pickle(save_name)
  print(f"DONE. Saved as {save_name}.")
