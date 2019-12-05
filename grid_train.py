import sys
import os
import nltk
import spacy
import gensim
import sklearn
import keras
import pandas as pd  
import numpy as np
from itertools import chain
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
nltk.download('wordnet') # run once
nltk.download('stopwords') # run once
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
import pyLDAvis.gensim
from keras.preprocessing.text import text_to_word_sequence
from sklearn.feature_extraction import stop_words
import subprocess
import shlex
import matplotlib.pyplot as plt
# %matplotlib inline

def default_stop():
  # intersection of gensim, nltk, spacy, and sklearn stopword lists
  default = ['me', 'inc', 'shan', "needn't", 'she', '‘s', 'therefore', 'find', 'down', 'thereupon', 'without', 'up', 'yourselves', 'many', 'eleven', 'full', 'de', 're', 'wherever', 'on', 'her', 'already', 'through', 'side', 'having', 'together', 't', 'take', "'m", 'therein', 'everyone', 'himself', 'whenever', 'them', "'s", 'once', 'forty', 'only', 'must', 'hereupon', 'moreover', 'my', 'very', 'say', 'whom', 'get', 'eg', 'does', 'll', 'indeed', 'everything', 'couldnt', '’m', 'not', 'each', 'using', 'do', 've', 'cant', 'if', 'various', 'throughout', 'otherwise', 'serious', 'd', 'regarding', 'mustn', 'yourself', 'noone', 'somewhere', 'twenty', 'most', 'thick', 'describe', 'however', 'fire', 'see', 'eight', 'while', 'besides', 'neither', 'well', 'us', 'below', 'is', "won't", 'might', 'mine', 'anywhere', 'weren', "'re", "n't", 'whereupon', 'becomes', 'should', 'hereafter', 'ours', 'during', 'a', 'ltd', 'con', 'isn', 'else', 'whither', 'shouldn', 'why', 'will', 'seems', 'ie', 'every', 'someone', 'bottom', 'ain', 'needn', 'then', 'thin', 'being', 'whereafter', 'via', 'never', 'same', "haven't", 'y', 'behind', 'name', 'give', 'move', 'some', 'six', 'we', 'whole', 'than', 'myself', 'our', "wasn't", 'now', 'whether', "mustn't", 'were', 'still', 'along', 'enough', 'for', 'yours', 'whereby', 'per', 'had', 'next', 'twelve', "doesn't", 'onto', 'cry', 'seeming', 'are', 'between', 'almost', 'third', 'latter', 'by', 'nevertheless', 'in', 'across', 'though', 'kg', 'somehow', 'out', 'show', 'no', 'either', 'didn', 'computer', '’ve', 'such', 'all', 'both', 'few', "weren't", 'from', '’d', 'doing', 'alone', 'nan', 'latterly', 's', 'although', 'fifteen', 'hasn', 'own', 'due', 'whereas', 'beyond', "you'd", "shouldn't", 'whose', 'who', 'n’t', 'unless', 'something', "shan't", 'other', 'also', 'they', 'make', 'three', 'been', 'found', 'whoever', 'doesn', 'first', 'made', 'ten', 'seem', '‘ll', 'of', 'your', 'at', 'the', 'where', 'further', 'has', 'former', 'their', 'or', 'four', 'so', 'wherein', 'empty', 'among', 'mill', 'be', 'hasnt', 'used', 'go', 'amongst', 'everywhere', 'fifty', "hadn't", '’ll', 'you', 'km', 'others', 'this', 'thru', 'may', 'wouldn', 'itself', "'d", 'please', 'could', 'done', 'several', 'afterwards', 'two', 'becoming', 'those', '‘ve', 'part', 'hundred', 'system', 'upon', "wouldn't", 'meanwhile', 'thus', '’s', 'herein', 'hadn', 'put', 'toward', 'hers', 'these', 'sometime', 'don', 'nine', 'have', 'won', 'least', 'thereafter', 'often', 'nobody', 'except', 'always', '’re', "you've", 'since', 'elsewhere', 'here', 'wasn', 'as', 'less', 'there', 'one', 'anyone', 'when', 'sometimes', 'its', 'formerly', 'ca', 'thence', 'm', "don't", 'rather', 'but', 'above', 'themselves', 'his', 'haven', 'what', 'too', 'aren', 'keep', "mightn't", 'top', 'he', 'anyhow', 'co', 'around', 'etc', 'about', 'nor', 'anyway', 'hence', '‘d', 'sixty', 'mostly', 'detail', 'anything', 'bill', 'much', "she's", 'ourselves', 'fify', 'that', 'last', 'theirs', 'really', 'back', 'un', 'yet', 'just', 'was', 'an', 'ma', "isn't", "you'll", "should've", 'until', 'off', 'perhaps', 'beside', 'nowhere', 'mightn', 'sincere', "'ll", "didn't", "it's", 'am', 'again', 'even', 'which', 'front', 'can', 'within', "aren't", 'him', "you're", 'and', 'namely', 'against', '‘re', "that'll", 'with', 'whence', 'five', 'amount', 'o', 'quite', 'call', 'interest', 'none', 'before', 'fill', 'how', 'it', 'ever', 'seemed', 'i', 'because', 'thereby', 'would', '‘m', 'couldn', "couldn't", 'did', "'ve", 'under', 'after', 'more', 'become', 'nothing', 'herself', 'to', 'any', 'over', 'into', "hasn't", 'hereby', 'towards', 'amoungst', 'whatever', 'became', 'n‘t', 'beforehand', 'another', 'cannot']
  default = list(set(default))
  return default

def custom_stop():
  default = default_stop()
  additional_words = []
  unstop = []
  custom = []
  for i in additional_words:
    default.append(i)
  for i in default:
    if i not in unstop:
      custom.append(i)
  custom = list(set(custom))
  return custom

def minimal_stop():
  default = default_stop()
  unstop = []
  minimal = []
  for i in default:
    if i not in unstop:
      minimal.append(i)
  minimal = list(set(minimal))
  return minimal

def pickle_df(df, pname):
  df.to_pickle(pname)
  print(f'Saved dataframe as "{pname}".')

def unpickle_df(pname, df):
  new_df = pd.read_pickle(pname)
  print(f'Loaded dataframe from "{pname}".')
  return new_df

def load_corpus(pkl):
  corpus = pd.read_pickle(pkl)
  corpus = corpus['tokens']
  return corpus

# build dictionary
def build_dict(corpus, no_below, no_above, keep_n):
  dictionary = gensim.corpora.Dictionary(corpus)
  dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
  return dictionary

# build Bag-of-Words corpus
def build_bow(corpus, dictionary):
  bow_corpus = [dictionary.doc2bow(doc) for doc in corpus]
  return bow_corpus

# build Term Frequency, Inverse Document Frequency corpus (TF-IDF)
def build_tfidf(bow_corpus):
  tfidf = models.TfidfModel(bow_corpus)
  corpus_tfidf = tfidf[bow_corpus]
  return corpus_tfidf

# print topics
def show_topics(n_topics, lda_model):
  print(f'Built LDA/TF-IDF model with {n_topics} topics: \n')
  for idx, topic in lda_model.print_topics():
    print(f'Topic #{idx}: {topic}')

# build topic model
def train_lda(corpus, dictionary, n_workers, n_passes, n_topics):
  # print("Building Bag-of-Words corpus...")
  bow_corpus = build_bow(corpus, dictionary)
  # print("Building TF-IDF corpus...")
  corpus_tfidf = build_tfidf(bow_corpus)  
  n_features = len(list(dictionary.values()))
  print(f'Training model with {n_topics} topics, {n_passes} passes, and {n_features} features...')
  lda_model = gensim.models.LdaMulticore(corpus_tfidf, num_topics=n_topics, id2word=dictionary, passes=n_passes, workers=n_workers)
  return lda_model

# save the trained model
def subprocess_cmd(command):
  process = subprocess.Popen(command,stdout=subprocess.PIPE, shell=True)
  proc_stdout = process.communicate()[0].strip()
  print(proc_stdout)

def save_model(lda_model, dictionary):
  print("Saving trained model...")
  filename_model = 'tf-lda.model'
  filename_dict = 'tf-lda.dict'
  lda_model.save(filename_model)
  dictionary.save(filename_dict)
  subprocess_cmd('bash tar_model.sh')
  # print("Model and dictionary saved to ./models/saved-model")

# evaluate topic model metrics of cohesion and log perplexity
def perplex(lda_model, corpus):
  perplexity = lda_model.log_perplexity(corpus)
  return perplexity

def cohere_umass(lda_model, corpus, dictionary):
  coherence_model_lda = CoherenceModel(model=lda_model, corpus=corpus, dictionary=dictionary, coherence='u_mass')
  coherence_lda = coherence_model_lda.get_coherence()
  return coherence_lda

def cohere_cv(lda_model, texts, dictionary):
  coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
  coherence_lda = coherence_model_lda.get_coherence()
  return coherence_lda

# train and save a single model
def train_single(corpus_filename, n_topics, no_below, no_above, n_passes, n_workers, keep_n):
  corpus = load_corpus(corpus_filename)
  dictionary = build_dict(corpus, no_below, no_above, keep_n)
  lda_model = train_lda(corpus, dictionary, n_workers, n_passes, n_topics)
  save_model(lda_model, dictionary)
  print("Training complete.")

# evaluate a single saved (pre-trained) model
def eval_single(corpus_filename, n_topics, no_below, no_above):
  processed_docs = pd.read_pickle(corpus_filename)
  processed_docs = processed_docs['tokens']
  filename_model = 'tf-lda.model'
  model = gensim.models.LdaModel.load(filename_model)
  filename_dict = 'tf-lda.dict'
  dictionary = corpora.Dictionary.load(filename_dict)
  bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
  tfidf = models.TfidfModel(bow_corpus)
  corpus_tfidf = tfidf[bow_corpus]
  n_features = len(list(dictionary.values()))
  print(f'Evaluating model trained with {n_topics} topics, n_passes={n_passes}, no_below={no_below}, no_above={no_above}, and {n_features} features...')
  print("Calculating Log Perplexity...")
  p_score = perplex(model, corpus_tfidf)
  print("Calculating Coherence (u_mass)...")
  c_umass_score = cohere_umass(model, corpus_tfidf, dictionary)
  print("Calculating Coherence (c_v)...")
  c_cv_score = cohere_cv(model, processed_docs, dictionary)
  # RESULTS FOR THIS MODEL
  results = (p_score, c_umass_score, c_cv_score)
  return results

# perform grid training and compare using evaluation metrics
def grid_train(corpus_filename, n_topics, no_below, no_above, n_passes, n_workers, keep_n):
  corpus = load_corpus(corpus_filename)
  dictionary = build_dict(corpus, no_below, no_above, keep_n)
  lda_model = train_lda(corpus, dictionary, n_workers, n_passes, n_topics)
  save_model(lda_model, dictionary)

  processed_docs = pd.read_pickle(corpus_filename)
  processed_docs = processed_docs['tokens']
  filename_model = 'tf-lda.model'
  model = gensim.models.LdaModel.load(filename_model)
  filename_dict = 'tf-lda.dict'
  dictionary = corpora.Dictionary.load(filename_dict)
  bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
  tfidf = models.TfidfModel(bow_corpus)
  corpus_tfidf = tfidf[bow_corpus]
  n_features = len(list(dictionary.values()))
  print(f'Evaluating model trained with {n_topics} topics, n_passes={n_passes}, no_below={no_below}, no_above={no_above}, and {n_features} features...')
  print("Calculating Log Perplexity...")
  p_score = perplex(model, corpus_tfidf)
  print("Calculating Coherence (u_mass)...")
  c_umass_score = cohere_umass(model, corpus_tfidf, dictionary)
  print("Calculating Coherence (c_v)...")
  c_cv_score = cohere_cv(model, processed_docs, dictionary)
  
  # RESULTS FOR THIS TRIAL
  results = (p_score, c_umass_score, c_cv_score)
  return results

if __name__ == '__main__':

  # set stopwords
  my_stopwords = default_stop()
  # custom_stopwords = custom_stop()
  # minimal_stopwords = minimal_stop()

  grid_results = []
  trained_topics = []
  trained_passes = []
  trained_no_below = []

  # import training data
  corpus_filename = input("\nEnter name of training corpus including 'tokens' column with preprocessed text [.pkl]: ")

  # define range of number of topics to use for grid training
  topics_start = int(input("Enter lowest number of topics for model training, e.g. '3': "))
  topics_end = int(input("Enter highest number of topics for model training, e.g. '20': ")) 
  range_n_topics = range(topics_start, topics_end + 1)

  # get values for other hyperparameters
  nb = int(input("Enter value for `no_below` [terms must appear in >= {n} docs], e.g. '30': "))
  no_below = nb
  na = float(input("Enter value for `no_above` [terms must not appear in > {0.n}->{n}% of docs], e.g. '0.5' (50%): "))
  no_above = na
  nps = int(input("Enter value for `n_passes` [perform {n} passes over the corpus], e.g. '5': "))
  n_passes = nps
  nwk = int(input("Enter value for `n_workers` [use {n} CPUs], e.g. '2': "))
  n_workers = nwk
  kpn = int(input("Enter value for `keep_n` [max {n} features for dict], e.g. '100000': "))
  keep_n = kpn

  # print selected hyperparameters
  print("\nHYPERPARAMETERS SELECTED: ")
  print(f'range_n_topics = range({topics_start}, {topics_end+1})')
  print(f'no_below = {no_below}')
  print(f'no_above = {no_above}')
  print(f'n_passes = {n_passes}')
  print(f'n_workers = {n_workers}')
  print(f'keep_n = {keep_n}')

  # begin grid training
  print("\nInitializing grid training...\n")
  for i in range_n_topics:
  # for i in range_n_passes:
  # for i in range_no_below:
  # for i in range_no_above:
    n_topics = i
    # no_above = i
    # no_below = i
    # n_passes = i
    trial = grid_train(corpus_filename, n_topics, no_below, no_above, n_passes, n_workers, keep_n)
    print(f'RESULTS: {trial}\n')
    grid_results.append(trial)
    trained_topics.append(i)
    trained_passes.append(n_passes)

  # print results
  print("GRID SEARCH COMPLETE:")
  print(grid_results)

  # analyze results
  trained_params = [x for x in range_n_topics]

  for x, y in zip(trained_params, grid_results):
    print(f'param_value={x}; results: {y}')

  print()

  x_params = []
  y_results = []
  za = []
  zb = []
  zc = []

  # weigh and scale results from each metric for visualization
  for x, y in zip(trained_params, grid_results):
    a = abs(y[0]) / 16
    b = abs(y[1]) / 4
    c = abs(y[2])
    calc = sum([a, b, c])/3
    x_params.append(x)
    y_results.append(calc)
    za.append(a)
    zb.append(b)
    zc.append(c)

  x = x_params
  y = y_results

  # display plot of evaluation metrics by parameter type (default: n_topics)
  plt.figure(figsize=(26,8))
  plt.plot(x,y, label='aggregate scores', linewidth=4)
  plt.plot(x,za, label='perplexity/16')
  plt.plot(x,zb, label='u_mass/4')
  plt.plot(x,zc, label='c_v')
  plt.title('Topic Model Coherence & Perplexity')
  plt.xlabel('Parameters Tested')
  plt.ylabel('Scores')
  plt.grid(True)
  plt.legend(loc=4)
  plt.show()

  print()

  for a, b, c in zip(x_params, zc, y_results):
    if b == min(zc):
      print(f'best cv_score = param_value: {a}, score = {b}')
    if c == min(y_results):
      print(f'best agg_score = param_value: {a}, score = {c}')

  print()
