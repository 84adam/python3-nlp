from flask import Flask, request
import os
import json
from random import randint
import calendar, datetime
import gensim
import keras
import pandas as pd  
import numpy as np
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
# nltk.download('wordnet')
from gensim.utils import simple_preprocess
from gensim import corpora, models
from keras.preprocessing.text import text_to_word_sequence

# define stopwords
def default_stop():
  # intersection of gensim, nltk, spacy, and sklearn stopword lists
  default = ['me', 'inc', 'shan', "needn't", 'she', '‘s', 'therefore', 'find', 'down', 'thereupon', 'without', 'up', 'yourselves', 'many', 'eleven', 'full', 'de', 're', 'wherever', 'on', 'her', 'already', 'through', 'side', 'having', 'together', 't', 'take', "'m", 'therein', 'everyone', 'himself', 'whenever', 'them', "'s", 'once', 'forty', 'only', 'must', 'hereupon', 'moreover', 'my', 'very', 'say', 'whom', 'get', 'eg', 'does', 'll', 'indeed', 'everything', 'couldnt', '’m', 'not', 'each', 'using', 'do', 've', 'cant', 'if', 'various', 'throughout', 'otherwise', 'serious', 'd', 'regarding', 'mustn', 'yourself', 'noone', 'somewhere', 'twenty', 'most', 'thick', 'describe', 'however', 'fire', 'see', 'eight', 'while', 'besides', 'neither', 'well', 'us', 'below', 'is', "won't", 'might', 'mine', 'anywhere', 'weren', "'re", "n't", 'whereupon', 'becomes', 'should', 'hereafter', 'ours', 'during', 'a', 'ltd', 'con', 'isn', 'else', 'whither', 'shouldn', 'why', 'will', 'seems', 'ie', 'every', 'someone', 'bottom', 'ain', 'needn', 'then', 'thin', 'being', 'whereafter', 'via', 'never', 'same', "haven't", 'y', 'behind', 'name', 'give', 'move', 'some', 'six', 'we', 'whole', 'than', 'myself', 'our', "wasn't", 'now', 'whether', "mustn't", 'were', 'still', 'along', 'enough', 'for', 'yours', 'whereby', 'per', 'had', 'next', 'twelve', "doesn't", 'onto', 'cry', 'seeming', 'are', 'between', 'almost', 'third', 'latter', 'by', 'nevertheless', 'in', 'across', 'though', 'kg', 'somehow', 'out', 'show', 'no', 'either', 'didn', 'computer', '’ve', 'such', 'all', 'both', 'few', "weren't", 'from', '’d', 'doing', 'alone', 'nan', 'latterly', 's', 'although', 'fifteen', 'hasn', 'own', 'due', 'whereas', 'beyond', "you'd", "shouldn't", 'whose', 'who', 'n’t', 'unless', 'something', "shan't", 'other', 'also', 'they', 'make', 'three', 'been', 'found', 'whoever', 'doesn', 'first', 'made', 'ten', 'seem', '‘ll', 'of', 'your', 'at', 'the', 'where', 'further', 'has', 'former', 'their', 'or', 'four', 'so', 'wherein', 'empty', 'among', 'mill', 'be', 'hasnt', 'used', 'go', 'amongst', 'everywhere', 'fifty', "hadn't", '’ll', 'you', 'km', 'others', 'this', 'thru', 'may', 'wouldn', 'itself', "'d", 'please', 'could', 'done', 'several', 'afterwards', 'two', 'becoming', 'those', '‘ve', 'part', 'hundred', 'system', 'upon', "wouldn't", 'meanwhile', 'thus', '’s', 'herein', 'hadn', 'put', 'toward', 'hers', 'these', 'sometime', 'don', 'nine', 'have', 'won', 'least', 'thereafter', 'often', 'nobody', 'except', 'always', '’re', "you've", 'since', 'elsewhere', 'here', 'wasn', 'as', 'less', 'there', 'one', 'anyone', 'when', 'sometimes', 'its', 'formerly', 'ca', 'thence', 'm', "don't", 'rather', 'but', 'above', 'themselves', 'his', 'haven', 'what', 'too', 'aren', 'keep', "mightn't", 'top', 'he', 'anyhow', 'co', 'around', 'etc', 'about', 'nor', 'anyway', 'hence', '‘d', 'sixty', 'mostly', 'detail', 'anything', 'bill', 'much', "she's", 'ourselves', 'fify', 'that', 'last', 'theirs', 'really', 'back', 'un', 'yet', 'just', 'was', 'an', 'ma', "isn't", "you'll", "should've", 'until', 'off', 'perhaps', 'beside', 'nowhere', 'mightn', 'sincere', "'ll", "didn't", "it's", 'am', 'again', 'even', 'which', 'front', 'can', 'within', "aren't", 'him', "you're", 'and', 'namely', 'against', '‘re', "that'll", 'with', 'whence', 'five', 'amount', 'o', 'quite', 'call', 'interest', 'none', 'before', 'fill', 'how', 'it', 'ever', 'seemed', 'i', 'because', 'thereby', 'would', '‘m', 'couldn', "couldn't", 'did', "'ve", 'under', 'after', 'more', 'become', 'nothing', 'herself', 'to', 'any', 'over', 'into', "hasn't", 'hereby', 'towards', 'amoungst', 'whatever', 'became', 'n‘t', 'beforehand', 'another', 'cannot']
  return default
my_stopwords = default_stop()

# generate random query id
def rand_id(n):
  n_digit_str = ''.join(["{}".format(randint(0, 9)) for num in range(0, n)])
  return int(n_digit_str)

# timestamp search/prediction
def get_ts():
  d = datetime.datetime.utcnow()
  ts = calendar.timegm(d.timetuple())
  return ts

# preprocessing functions
stemmer = PorterStemmer()

def lemmatize_stemming(text):
  return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
  
def preprocess(text):
  result = []
  for token in gensim.utils.simple_preprocess(text):
    if token not in my_stopwords and len(token) > 2:
      result.append(lemmatize_stemming(token))
  return result

def word_split(doc):
  words = []
  for word in doc.split(' '):
    words.append(word)
  return words

# infer topics from query and produce JSON object as output
def gen_json(doc):
  query_text = doc
  tokens = preprocess(doc)
  vector = infer_topic(tokens)
  source = "Search Bar"
  query_id = rand_id(10)
  query_ts = get_ts()
  q_dict = ({'source': f'{source}', 'query_id': f'{query_id}', 'query_ts': f'{query_ts}', 'query_text': f'{query_text}', 'tokens': f'{tokens}', 'topics': f'{vector}'})
  return json.dumps(q_dict)
 
def infer_topic(tokens):
  dict_new = dictionary.doc2bow(tokens)
  vector = model[dict_new]
  return vector

def load_model():
  filepath = os.getcwd()  
  filename_model = filepath + '/' + 'tf-lda.model'
  filename_dict = filepath + '/' + 'tf-lda.dict'
  model = gensim.models.LdaModel.load(filename_model)
  dictionary = corpora.Dictionary.load(filename_dict)
  return model, dictionary

app = Flask(__name__)

#create a 'search' route
@app.route('/search', methods=['POST', 'GET'])

# create search form, submitted text processed and returned with inferred topics
def search():
  if request.method == 'POST':
    query = request.form.get('query')
    new_json = gen_json(query)
    return f'''<h4>You entered: "<i>{query}</i>"</h4>
               <h4>Results (JSON):</h4>
               <p>{new_json}</p>
               <h4>List of Model Topics:</h4>
               <p><font size="2">{model_topics}</font></p>'''
  return '''<form method="POST">
  Search Terms: <input type="text" name="query">
  <input type="submit">
  </form>'''

if __name__ == '__main__':
  # load model and model dictionary files
  topic_model = load_model()
  model = topic_model[0]
  dictionary = topic_model[1]
  model_topics = []
  for i in range(0, model.num_topics):
    model_topics.append(f'Topic #{i}: {model.print_topic(i)}')
  model_topics = '<br>'.join([f'{str(x)}' for x in model_topics])
  
  # start server, await user queries
  app.run(debug=True, host="127.0.0.1", port=5000)
