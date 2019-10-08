import sys
import os
import nltk
import spacy
import gensim
import sklearn
import keras
import pandas as pd  
import numpy as np
from scipy.spatial import distance
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
nltk.download('wordnet')
nltk.download('stopwords')
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess
from gensim import corpora, models
from keras.preprocessing.text import text_to_word_sequence
from sklearn.feature_extraction import stop_words

# define stopwords

def add_words(filename):
    with open(filename) as f:
        additional_words = f.readlines()
    additional_words = [x.strip() for x in additional_words]
    return additional_words

def remove_words(filename):
    with open(filename) as f:
        unstop = f.readlines()
    unstop = [x.strip() for x in unstop]
    return unstop

def define_stopwords():
	"""
	default: combine SKLEARN, NLTK, and SPACY stopwords -> 'sns_set'
	alternative: set custom 'additional_words' and 'unstop' words (to ignore)
	function returns a list: 'stopwords'
	"""

	# corpus-specific stop words [OPTIONAL]
        # add 'stop.txt' to local directory, pass as argument 2
	additional_words = ['nan']

	# don't remove these words which may be important in our context [OPTIONAL]
        # add 'unstop.txt' to local directory, pass as argument 3
	unstop = []

	gen_stop = gensim.parsing.preprocessing.STOPWORDS
	nlp = spacy.load('en')
	spacy_stop = nlp.Defaults.stop_words # .add("my_new_stopword")
	sk_stop = stop_words.ENGLISH_STOP_WORDS
	nltk_stop = stopwords.words('english')

	custom_stop = additional_words

	sns_stop = []
	all_stop = []

	# combine sklearn, nltk, and spacy stop word lists: sns_stop
	# also add these to all_stop
	for i in gen_stop:
		if i not in unstop:
			sns_stop.append(i)
			all_stop.append(i)

	for i in spacy_stop:
		if i not in unstop:
			sns_stop.append(i)
			all_stop.append(i)

	for i in sk_stop:
		if i not in unstop:
			sns_stop.append(i)
			all_stop.append(i)

	for i in nltk_stop:
		if i not in unstop:
			sns_stop.append(i)
			all_stop.append(i)

	# add corpus specific stop words to all_stop
	for i in custom_stop:
		if i not in unstop:
			if i not in all_stop:
				all_stop.append(i)

	sns_set = list(set(sns_stop))
	all_set = list(set(all_stop))

	if len(custom_stop) == 0 and len(unstop) == 0:
		# print(f'sns_set stopwords = {len(sns_set)} words: \nExamples: \n{[x for x in sns_set[0:10]]}\n{[x for x in sns_set[10:20]]}')
		my_stopwords = sns_set
	else:
		# print(f'all_set (custom) stopwords = {len(all_set)} words: \nExamples: \n{[x for x in all_set[0:10]]}\n{[x for x in all_set[10:20]]}')
		my_stopwords = all_set

	return my_stopwords


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


# Infer Topic Probability Distribution of New Document

def infer_topic(new_doc):
    pre_new = preprocess(new_doc) # remove stop-words, lemmatize, and stem
    dict_new = dictionary.doc2bow(pre_new) # build term-frequency dictionary
    first_five = [f'{dict_new[i][0]}: \"{dictionary[dict_new[i][0]]}\"*{dict_new[i][1]}' for i in range(len(dict_new[0:5]))]
    vector = model[dict_new] # get topic probability distribution for new_doc
    return vector

# Compare two probability distributions using Jensen-Shannon Distance
# distance = sqrt(divergence)

def jsdist(p, q):
	return distance.jensenshannon(p, q, base=None)

# Interpolate missing topics and probability values (missing -> 0%)

def interp_jsdist(p, q):
	p_topics = [x[0] for x in p]
	q_topics = [x[0] for x in q]
	
	p_missing = list(set(q_topics).difference(p_topics))
	q_missing = list(set(p_topics).difference(q_topics))
	
	if len(p_missing) > 0:
		for i in p_missing:
			missing = (i, 0)
			p.append(missing)
	
	if len(q_missing) > 0:
		for i in q_missing:
			missing = (i, 0)
			q.append(missing)
	
	p = sorted(vec_A)
	q = sorted(vec_B)
	
	print("\nDistributions: ")
	print(p)
	print(q)
	print("\nDistance: ")
	
	pq_dist = jsdist(p, q)[1]
	
	return pq_dist

# run program

if __name__ == '__main__':

    filepath = str(sys.argv[1]) # path to saved tf-lda* files, stop.txt, unstop.txt
    doc_A = str(sys.argv[2])
    doc_B = str(sys.argv[3])

    filename_model = filepath + '/' + 'tf-lda.model'
    filename_dict = filepath + '/' + 'tf-lda.dict'
    filename_stop = filepath + '/' + 'stop.txt'
    filename_unstop = filepath + '/' + 'unstop.txt'

    my_stopwords = define_stopwords()
    new_words = add_words(filename_stop)
    new_unstop = remove_words(filename_unstop)
    for i in new_words:
        my_stopwords.append(i)
    my_stopwords = [w for w in my_stopwords if w not in new_unstop]
    my_stopwords = list(set(my_stopwords))

    model = gensim.models.LdaModel.load(filename_model)
    dictionary = corpora.Dictionary.load(filename_dict)

    # print all topics
    print()

    for i in range(0, model.num_topics):
        print(f'Topic #{i}: {model.print_topic(i)}')
   
    print(f'\nPerforming inference on documents...')
    vec_A = infer_topic(doc_A)
    vec_B = infer_topic(doc_B)
    print(interp_jsdist(vec_A, vec_B))
