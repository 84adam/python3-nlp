import os
import gensim
from gensim.utils import simple_preprocess
from gensim import corpora, models
import pandas as pd
import numpy as np
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary

# load pre-trained model and corpus (from current working directory)

def load_model():
  filepath = os.getcwd()  
  filename_model = filepath + '/' + 'tf-lda.model'
  filename_dict = filepath + '/' + 'tf-lda.dict'
  model = gensim.models.LdaModel.load(filename_model)
  dictionary = corpora.Dictionary.load(filename_dict)
  return model, dictionary

def load_corpus():
  processed_docs = pd.read_pickle('processed_docs.pkl')
  dictionary = load_model()[1]
  bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
  tfidf = models.TfidfModel(bow_corpus)
  corpus = tfidf[bow_corpus]
  return corpus

# calculate model perplexity and coherence scores
# for each a higher score is better
# e.g. for log perplexity: -13.5 is better than -15.5
# e.g. for coherence: -1.5 is better than -3.5

def perplex(lda_model, corpus):
  perplexity = lda_model.log_perplexity(corpus)
  return perplexity

def cohere(lda_model, corpus, dictionary):
  coherence_model_lda = CoherenceModel(model=lda_model, corpus=corpus, dictionary=dictionary, coherence='u_mass')
  coherence_lda = coherence_model_lda.get_coherence()
  return coherence_lda

if __name__ == '__main__':
  topic_model = load_model()
  lda_model = topic_model[0]
  dictionary = topic_model[1]
  corpus = load_corpus()
  p_score = perplex(lda_model, corpus)
  c_score = cohere(lda_model, corpus, dictionary)
  print(f'Model Log Perplexity Score: {p_score}')
  print(f'Model Coherence Score: {c_score}')
