import sys
import os
from flask import Flask, request
from pprint import pprint
import json
import nltk
import spacy
import gensim
import sklearn
import keras
import pandas as pd  
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
nltk.download('wordnet') # run once
nltk.download('stopwords') # run once
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess
from gensim import corpora, models
from keras.preprocessing.text import text_to_word_sequence
from sklearn.feature_extraction import stop_words
from scipy.spatial import distance
from random import randint
import calendar, datetime

"""
SEARCH_APP: Launch search engine.
Set host and port prior to running.
Requires a corpus or sub-corpus with inferred topic vectors, i.e.:

                                                title                                                raw                                             tokens                                             topics
0        https://en.wikipedia.org/wiki/Graphic_design  graphic design is the process of visual commun...  [graphic, design, process, visual, commun, pro...  [(0, 0.63671833), (1, 0.0), (2, 0.0), (3, 0.29...
1        https://en.wikipedia.org/wiki/Design_fiction  design fiction is a design practice aiming at ...  [design, fiction, design, practic, aim, explor...  [(0, 0.9217787), (1, 0.0), (2, 0.0), (3, 0.076...
2   https://en.wikipedia.org/wiki/Creativity_techn...  creativity techniques are methods that encoura...  [creativ, techniqu, method, encourag, creativ,...  [(0, 0.9970473), (1, 0.0), (2, 0.0), (3, 0.0),...
3        https://en.wikipedia.org/wiki/Jewelry_design  jewellery design is the art or profession of d...  [jewelleri, design, art, profess, design, crea...  [(0, 0.80666345), (1, 0.0), (2, 0.18880607), (...
4     https://en.wikipedia.org/wiki/Benjamin_Franklin  benjamin franklin frs frsa frse january 17 170...  [benjamin, franklin, fr, frsa, frse, januari, ...  [(0, 0.9998033), (1, 0.0), (2, 0.0), (3, 0.0),...
5      https://en.wikipedia.org/wiki/Strategic_design  strategic design is the application of future ...  [strateg, design, applic, futur, orient, desig...  [(0, 0.45011556), (1, 0.0), (2, 0.0), (3, 0.54...
6   https://en.wikipedia.org/wiki/Activity-centere...  activity centered design acd is an extension o...  [activ, center, design, acd, extens, human, ce...  [(0, 0.6329251), (1, 0.0), (2, 0.0), (3, 0.344...
7          https://en.wikipedia.org/wiki/Architecture  architecture latin architectura from the greek...  [architectur, latin, architectura, greek, ἀρχι...  [(0, 0.9993874), (1, 0.0), (2, 0.0), (3, 0.0),...
8         https://en.wikipedia.org/wiki/Web_developer  a web developer is a programmer who specialize...  [web, develop, programm, special, specif, enga...  [(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.8699879),...
9   https://en.wikipedia.org/wiki/Sonic_interactio...  sonic interaction design is the study and expl...  [sonic, interact, design, studi, exploit, soun...  [(0, 0.8485447), (1, 0.0), (2, 0.0), (3, 0.0),...
10       https://en.wikipedia.org/wiki/Costume_design  costume design is the investing of clothing an...  [costum, design, invest, cloth, overal, appear...  [(0, 0.9970691), (1, 0.0), (2, 0.0), (3, 0.0),...
11  https://en.wikipedia.org/wiki/Software_applica...  application software app for short is software...  [applic, softwar, app, short, softwar, design,...  [(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.9974447),...
12          https://en.wikipedia.org/wiki/Art_Nouveau  art nouveau ˌɑːrt nuːˈvoʊ ˌɑːr french   aʁ nuv...  [art, nouveau, ˌɑːrt, nuːˈvoʊ, ˌɑːr, french, n...  [(0, 0.9998343), (1, 0.0), (2, 0.0), (3, 0.0),...
13  https://en.wikipedia.org/wiki/Philosophy_of_de...  philosophy of design is the study of definitio...  [philosophi, design, studi, definit, design, a...  [(0, 0.9634965), (1, 0.0), (2, 0.0), (3, 0.0),...
14  https://en.wikipedia.org/wiki/Environmental_im...  environmental impact design eid is the design ...  [environment, impact, design, eid, design, dev...  [(0, 0.67384595), (1, 0.3187163), (2, 0.0), (3...

This serves as the pool of candidate results.

Query text topics are derived from the pre-loaded model.

The distance between the query's topic probability distribution and that of each of the candidate documents is measured using Jensen-Shannon Distance.

The nearest 1% of candidate documents are returned as results to the user.

These are returned in rank order from closest to furthest in terms of JSD, where the closest is 1.0 and the furthest is 0.0.

This returns to the user the document with the highest relevance in terms of 'topic profile' from among the available candidate documents.

"""

# generate random query id

def rand_id(n):
    n_digit_str = ''.join(["{}".format(randint(0, 9)) for num in range(0, n)])
    return int(n_digit_str)

# timestamp predictions

def get_ts():
    d = datetime.datetime.utcnow()
    ts = calendar.timegm(d.timetuple())
    return ts

# define stopwords
def default_stop():
    # intersection of gensim, nltk, spacy, and sklearn stopword lists
    default = ['me', 'inc', 'shan', "needn't", 'she', '‘s', 'therefore', 'find', 'down', 
	'thereupon', 'without', 'up', 'yourselves', 'many', 'eleven', 'full', 'de', 're', 
	'wherever', 'on', 'her', 'already', 'through', 'side', 'having', 'together', 't', 
	'take', "'m", 'therein', 'everyone', 'himself', 'whenever', 'them', "'s", 'once', 
	'forty', 'only', 'must', 'hereupon', 'moreover', 'my', 'very', 'say', 'whom', 'get', 
	'eg', 'does', 'll', 'indeed', 'everything', 'couldnt', '’m', 'not', 'each', 'using', 
	'do', 've', 'cant', 'if', 'various', 'throughout', 'otherwise', 'serious', 'd', 
	'regarding', 'mustn', 'yourself', 'noone', 'somewhere', 'twenty', 'most', 'thick', 
	'describe', 'however', 'fire', 'see', 'eight', 'while', 'besides', 'neither', 'well', 
	'us', 'below', 'is', "won't", 'might', 'mine', 'anywhere', 'weren', "'re", "n't", 
	'whereupon', 'becomes', 'should', 'hereafter', 'ours', 'during', 'a', 'ltd', 'con', 
	'isn', 'else', 'whither', 'shouldn', 'why', 'will', 'seems', 'ie', 'every', 'someone', 
	'bottom', 'ain', 'needn', 'then', 'thin', 'being', 'whereafter', 'via', 'never', 
	'same', "haven't", 'y', 'behind', 'name', 'give', 'move', 'some', 'six', 'we', 
	'whole', 'than', 'myself', 'our', "wasn't", 'now', 'whether', "mustn't", 'were', 
	'still', 'along', 'enough', 'for', 'yours', 'whereby', 'per', 'had', 'next', 'twelve', 
	"doesn't", 'onto', 'cry', 'seeming', 'are', 'between', 'almost', 'third', 'latter', 
	'by', 'nevertheless', 'in', 'across', 'though', 'kg', 'somehow', 'out', 'show', 'no', 
	'either', 'didn', 'computer', '’ve', 'such', 'all', 'both', 'few', "weren't", 'from', 
	'’d', 'doing', 'alone', 'nan', 'latterly', 's', 'although', 'fifteen', 'hasn', 'own', 
	'due', 'whereas', 'beyond', "you'd", "shouldn't", 'whose', 'who', 'n’t', 'unless', 
	'something', "shan't", 'other', 'also', 'they', 'make', 'three', 'been', 'found', 
	'whoever', 'doesn', 'first', 'made', 'ten', 'seem', '‘ll', 'of', 'your', 'at', 'the', 
	'where', 'further', 'has', 'former', 'their', 'or', 'four', 'so', 'wherein', 'empty', 
	'among', 'mill', 'be', 'hasnt', 'used', 'go', 'amongst', 'everywhere', 'fifty', 
	"hadn't", '’ll', 'you', 'km', 'others', 'this', 'thru', 'may', 'wouldn', 'itself', 
	"'d", 'please', 'could', 'done', 'several', 'afterwards', 'two', 'becoming', 'those', 
	'‘ve', 'part', 'hundred', 'system', 'upon', "wouldn't", 'meanwhile', 'thus', '’s', 
	'herein', 'hadn', 'put', 'toward', 'hers', 'these', 'sometime', 'don', 'nine', 'have', 
	'won', 'least', 'thereafter', 'often', 'nobody', 'except', 'always', '’re', "you've", 
	'since', 'elsewhere', 'here', 'wasn', 'as', 'less', 'there', 'one', 'anyone', 'when', 
	'sometimes', 'its', 'formerly', 'ca', 'thence', 'm', "don't", 'rather', 'but', 'above', 
	'themselves', 'his', 'haven', 'what', 'too', 'aren', 'keep', "mightn't", 'top', 'he', 
	'anyhow', 'co', 'around', 'etc', 'about', 'nor', 'anyway', 'hence', '‘d', 'sixty', 
	'mostly', 'detail', 'anything', 'bill', 'much', "she's", 'ourselves', 'fify', 'that', 
	'last', 'theirs', 'really', 'back', 'un', 'yet', 'just', 'was', 'an', 'ma', "isn't", 
	"you'll", "should've", 'until', 'off', 'perhaps', 'beside', 'nowhere', 'mightn', 
	'sincere', "'ll", "didn't", "it's", 'am', 'again', 'even', 'which', 'front', 'can', 
	'within', "aren't", 'him', "you're", 'and', 'namely', 'against', '‘re', "that'll", 
	'with', 'whence', 'five', 'amount', 'o', 'quite', 'call', 'interest', 'none', 'before', 
	'fill', 'how', 'it', 'ever', 'seemed', 'i', 'because', 'thereby', 'would', '‘m', 
	'couldn', "couldn't", 'did', "'ve", 'under', 'after', 'more', 'become', 'nothing', 
	'herself', 'to', 'any', 'over', 'into', "hasn't", 'hereby', 'towards', 'amoungst', 
	'whatever', 'became', 'n‘t', 'beforehand', 'another', 'cannot']
    return default

my_stopwords = default_stop()

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

def flatten_text(doc):
    output = ' '.join([w for w in text_to_word_sequence(str(doc))])
    return output

def gen_dict_vector(doc):
    query_text = doc
    tokens = preprocess(doc)
    vector = interp_topics(infer_topic(tokens))
    source = "Search Bar"
    query_id = rand_id(10)
    query_ts = get_ts()
    q_dict = ({'source': f'{source}', 'query_id': f'{query_id}', 'query_ts': 
    f'{query_ts}', 'query_text': f'{query_text}', 'tokens': f'{tokens}', 'topics': f'{vector}'})
    return q_dict, vector

def gen_json(q_dict):
    return json.dumps(q_dict)

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

def load_compare_docs(pkl_filename):
    compare_docs = pkl_filename
    tdf = unpickle_df(compare_docs, 'tdf')
    tt = tdf['title']
    rw = tdf['raw']
    tp = tdf['topics']
    return tt, rw, tp

def gen_json_results(vector, compare_docs, thresh):
    r_titles = compare_docs[0]
    r_raw = compare_docs[1]
    r_topics = compare_docs[2]
    r_distances = all_jsd(vector, r_topics) # measure JSD between vector and all compare_docs
    rdf = pd.DataFrame({'title': [x for x in r_titles], 'raw': [x for x in r_raw], 
    'topics': [x for x in r_topics], 'distances': [x for x in r_distances]})
    tt = rdf['title']
    rw = rdf['raw']
    tp = rdf['topics']
    aj = rdf['distances']
    pct_val = thresh
    pct_thresh = np.percentile(aj, pct_val)
    filtered = rdf[rdf['distances'] <= pct_thresh]
    filtered = filtered.sort_values(by=['distances'])
    tt = filtered['title']
    rw = filtered['raw']
    tp = filtered['topics']
    aj = filtered['distances']
    def confidence(n):
    	pct = abs(n-1)*100
    	return pct
    ajc = aj.map(confidence)
    rwf = rw.map(flatten_text)
    # sort and jsonify results
    results_df = pd.DataFrame({'title': [x for x in tt], 'score': [f'{x:.0f}' for x in ajc], 'text': [x[0:500] for x in rwf], 'topics': [x for x in tp]})
    results_df = results_df.sort_values(by=['score'], ascending=[False])
    r_json = results_df.to_json(orient='index')
    return r_json, results_df

def format_results(df):
    tt = df['title']
    rw = df['text']
    tp = df['topics']
    aj = df['score'].apply(float)
    output = []
    for a, b, c, d in zip(tt, rw, tp, aj):
        result = f'{abs(d-1)*1:.0f}% match - URL: <a href="{a}">{a}</a> - SNIPPET: {b[0:200]}...'
        output.append(result)
    return output

#### WEB APP ####
app = Flask(__name__)
#search route
@app.route('/search', methods=['POST', 'GET'])

def search():
    if request.method == 'POST':
        query = request.form.get('query')
        # run inference on query
        inferred = gen_dict_vector(query)
        query_json = gen_json(inferred[0])
        vector = inferred[1]
        results = gen_json_results(vector, compare_docs, thresh)
        formatted = format_results(results[1])
        return f'''<h4>You entered: "<i>{query}</i>"</h4>
                   <h4>Query Analysis:</h4>
                   <p>{query_json}</p>
                   <h4>Best Matches:</h4>
                   <p>{' '.join([x + "<br>" for x in formatted])}</p>
                   <p>{results[0]}</p>'''
    return '''<form method="POST">
	Search Terms: <input type="text" name="query">
	<input type="submit">
	</form>'''

if __name__ == '__main__':
  
  candidate_documents = input("Enter name of corpus of candidate documents to use (.pkl): ")
  compare_docs = load_compare_docs(candidate_documents)
  thresh = 1 # Return top 1% of matches.
  
  print("Loading model...")
  num_topics = int(input("How many topics were used to train this model? "))

  topic_model = load_model()
  model = topic_model[0]
  dictionary = topic_model[1]

  # METHOD TO BUILD CANDIDATE DOCS CORPUS USING PRE-TRAINED MODEL:
  # corpus_filename = input("Enter name of corpus on which to model topic vectors: ")
  # save_name = input("Save candidate corpus as (enter a filename[.pkl]): ")
  # save_name = save_name + ".pkl"
  # model = gensim.models.LdaModel.load(filename_model)
  # dictionary = corpora.Dictionary.load(filename_dict)
  # n_features = len(list(dictionary.values()))
  # topics = []
  # for i in tokens:
  #   result = interp_topics(infer_topic(i))
  #   topics.append(result)
  # df = pd.read_pickle(corpus_filename)
  # title = df['title']
  # raw = df['raw']
  # df = pd.DataFrame({'title': [x for x in title], 'raw': [x for x in raw], 'tokens': [x for x in tokens], 'topics': [x for x in topics]})
  # df.to_pickle(save_name)
  # print(f"DONE. Saved as {save_name}.")

  # START SERVER & AWAIT USER INPUT
  app.run(debug=True, host="127.0.0.1", port=5000)
