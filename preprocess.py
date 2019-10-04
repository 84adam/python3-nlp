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

# process corpus csv, dataframe, and start preprocessing

def sns_preprocess(filename):
	"""
	convert csv file to dataframe
	drop empty rows
	
	automatically select longest text column by average cell length in words
	... using 'df_field_col_len()' function
	
	preprocess only the longest text column
	return 'processed_docs'
	"""  
	corpus_filename = str(filename)
	df = pd.read_csv(corpus_filename)
	df = df.dropna()
	corpus = df_field_col_len(df)
	print("Initializing preprocessing...")
	processed_docs = corpus.map(preprocess)  
	print("Built: `processed_docs` Pandas Series object...")
	return processed_docs
    
def df_field_col_len(df):
	"""
	Compute average length of each field in data set
	on a per-column basis.
	
	Use this to determine which column in dataframe
	contains documents or text strings that can be 
	preprocessed and analyzed.
	"""
    
	# Get average length of each field (per column) in data set
	sum_i_list = [[] for i in range(len(list(df.columns.values)))]
	avg_len_i_list = [[] for i in range(len(list(df.columns.values)))]
	
	count = 0
	
	for i in list(df.columns.values):
		sum_i = sum([len(text_to_word_sequence(x)) for x in df[i].apply(str)])
		sum_i_list[count].append(sum_i)
		len_i = len(df[i])
		avg_len_i_list[count].append(sum_i / len_i)  
		count += 1
	
	for x, y in zip(list(df.columns.values), avg_len_i_list):
		print("Average Length of Field '{0:<15s}': \t{1:.2f}".format(x.replace("\n", ""), y[0]))
	
	# Locate longest average field length (per cell) from all columns
	
	for x, y in zip(list(df.columns.values), avg_len_i_list):
		if [y[0]] == max(avg_len_i_list):
			print("Longest average field length by word count found in '{0}' column (avg. {1:.0f} words).".format(x.replace("\n", ""), y[0]))
	
	longest = ''
	
	for x, y in zip(list(df.columns.values), avg_len_i_list):
		if [y[0]] == max(avg_len_i_list):
			longest = x
    
	print("Preprocessing '{}' column from corpus...".format(longest))
	
	df['index1'] = df.index
	
	doc_index = df['index1']
	
	corpus = df[longest]
  
	return corpus
  
# run program from command line given 1 argument == csv filename
# or ask for user input and run program

if __name__ == '__main__':
	
    try:
        filename = str(sys.argv[1])
    except Exception as e:
        print("ERROR: ", e)
        print("Please provide filename below [.csv].")
        try:
            exit
        except:
            sys.exit(2)
	
    try:
        filename
    except:
	
        try:
            filename = str(input("Please enter a filename [.csv]: "))
        except Exception as e:
            print("ERROR: ", e)
            exit
	
        if filename == '-f':
            try:
                 filename = str(input("Please enter a filename [.csv]: "))
            except Exception as e:
                print("ERROR: ", e)
                exit
	
    if filename in os.listdir():
        print(f'Processing {filename}...')
        processed_docs = sns_preprocess(filename)
        save_file = "processed_docs.pkl"
        processed_docs.to_pickle(save_file)
        print(f'Processing complete. Saved to {os.getcwd()}/{save_file}.')
        print(f'To re-load, use pandas.read_pickle("/PATH/TO/{save_file}")')
    else:
        print(f'File not found in {os.getcwd()} directory. Please check and try again.')
