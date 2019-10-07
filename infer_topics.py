import sys
import os
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
nltk.download('wordnet')
nltk.download('stopwords')
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess
from gensim import corpora, models
from keras.preprocessing.text import text_to_word_sequence
from sklearn.feature_extraction import stop_words

# define stopwords

def define_stopwords():
	"""
	default: combine SKLEARN, NLTK, and SPACY stopwords -> 'sns_set'
	alternative: set custom 'additional_words' and 'unstop' words (to ignore)
	function returns a list: 'stopwords'
	"""
	
	# corpus-specific stop words [OPTIONAL]
	additional_words = ['nan']
	
	# don't remove these words which may be important in our context
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
		print(f'sns_set stopwords = {len(sns_set)} words: \nExamples: \n{[x for x in sns_set[0:10]]}\n{[x for x in sns_set[10:20]]}')
		my_stopwords = sns_set
	else:
		print(f'all_set (custom) stopwords = {len(all_set)} words: \nExamples: \n{[x for x in all_set[0:10]]}\n{[x for x in all_set[10:20]]}')
		my_stopwords = all_set
	
	return my_stopwords
    
# stopwords
  
my_stopwords = define_stopwords()
  
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

# load trained model from file

filename_model = 'tf-lda.model'
filename_dict = 'tf-lda.dict'

print("Loading model...")

model = gensim.models.LdaModel.load(filename_model)
dictionary = corpora.Dictionary.load(filename_dict)

print(f'Loaded {filename_model} and {filename_dict}.\n')

# print all topics

for i in range(0, model.num_topics):
    print(f'Topic #{i}: {model.print_topic(i)}')
    
print("\n")

# Perform Topic Probability Distribution on an Unseen Document

def infer_topic(new_doc):
  print("\nAnalyzing new document:")
	print("(1) Performing preprocessing...")
	pre_new = preprocess(new_doc) # remove stop-words, lemmatize, and stem
	print(f'{pre_new[0:5]} ...')
	print("(2) Building term-frequency dictionary...")
	dict_new = dictionary.doc2bow(pre_new) # build term-frequency dictionary
	first_five = [f'{dict_new[i][0]}: \"{dictionary[dict_new[i][0]]}\"*{dict_new[i][1]}' for i in range(len(dict_new[0:5]))]
	print(f'{[x for x in first_five]}')
	print("(3) Inferring topic distribution...")
	vector = model[dict_new] # get topic probability distribution for new_doc
	print("Topic Probability Distribution:")
	print(vector)

if __name__ == '__main__':

	# If running in Jupyter Notebook, paste in a string of text
	# If running from the command line, instead save doc in a text file locally and do:
	# ... `$ python3 infer_topics.py "$(cat <textfile.txt>)"`
  # If you have troubles on the command line, try stripping out punctuation, etc.;
  # ... Use `clean.sh` and redirect to a new text file if needed.
	
	try:
		new_doc = str(sys.argv[1])
	except Exception as e:
		print("ERROR: ", e)
		exit
	
	try:
		new_doc = str(input("Please paste in a text string: "))
	except Exception as e:
		print("ERROR: ", e)

	infer_topic(new_doc)
