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
  # OR: add 'stop.txt' to local directory
	additional_words = []
	
	# don't remove these words which may be important in our context [OPTIONAL]
  # OR: add 'unstop.txt' to local directory
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
		my_stopwords = sns_set
	else:
		my_stopwords = all_set
	
	return my_stopwords
    
# preprocessing functions

stemmer = PorterStemmer()

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
  
def preprocess(text):
	result = []
	for token in gensim.utils.simple_preprocess(text, min_len=2, max_len=17):
		if token not in my_stopwords:
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
	return 'processed_docs' tokens series and dataframe as 'df'
	"""  
	corpus_filename = str(filename)
	df = pd.read_csv(corpus_filename)
	df = df.dropna()
	corpus = df_field_col_len(df)
	print("Initializing preprocessing...")
	processed_docs = corpus.map(preprocess)  
	print("Built: `processed_docs` Pandas Series object...")
	return processed_docs, df
    
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
  
if __name__ == '__main__':
  raw_docs = input("Enter name of .csv file to be processed: ")
  out_file_series = input("Save 'tokens' series as (enter filename, e.g. 'processed_docs.pkl'): ")
  out_file_df = input("Save corpus with new 'tokens' column as (enter filename, e.g. 'processed_df.pkl'): ")
  my_stopwords = define_stopwords()
  print(f'Default Stopwords List = {len(my_stopwords)} words.')
  print(f'Example: {[x for x in my_stopwords[0:5]]}.')
  print(f'Example: {[x for x in my_stopwords[20:26]]}.')
  print(f'Example: {[x for x in my_stopwords[-6:-1]]}.')
  
  print("\nIf you wish to ADD custom stop words, list them in a `stop.txt` file in the current directory.")
  print("If you wish to REMOVE stop words from the default set, list them in an `unstop.txt` file in the current directory.\n")
  
  if 'stop.txt' in os.listdir():
    print(f"Adding custom stopwords from 'stop.txt'...")
    new_words = add_words('stop.txt')
  else:
    print("No additional words to add, or no 'stop.txt' file found.")

  if 'unstop.txt' in os.listdir():
    print(f"Removing stopwords as per 'unstop.txt'...")
    new_unstop = remove_words('unstop.txt')
  else:
    print("No words to remove, or no 'unstop.txt' file found.")
    
  for i in new_words:
    my_stopwords.append(i)
    
  my_stopwords = [w for w in my_stopwords if w not in new_unstop]
  my_stopwords = list(set(my_stopwords))

  print(f'Total Stopwords = {len(my_stopwords)} words.')
  print(f'Example: {[x for x in my_stopwords[0:5]]}.')
  print(f'Example: {[x for x in my_stopwords[20:26]]}.')
  print(f'Example: {[x for x in my_stopwords[-6:-1]]}.')

  print(f'\nProcessing {raw_docs}...')
  processed_data = sns_preprocess(raw_docs)
  processed_docs = processed_data[0]
  processed_df = processed_data[1]
  processed_df = processed_df.assign(tokens=[x for x in processed_docs])
  
  # save tokens series object as .pkl
  processed_docs.to_pickle(out_file_series)

  # save corpus with tokens column as .pkl
  processed_df.to_pickle(out_file_df)

  # notify upon completion
  print(f'Processing complete. Saved to {os.getcwd()}: {out_file_series}, {out_file_df}.')
  print(f'To re-load, use pandas.read_pickle("/PATH/TO/<filename>.pkl")')
