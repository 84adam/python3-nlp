import os
import gensim
import pandas as pd  
import numpy as np
import sys
import subprocess
import shlex
from gensim import corpora, models

# build topic model

arguments = sys.argv[1:]

if len(arguments) < 7:
    print("\nERROR: Missing Required Arguments: ")
    print("(1) dict_no_below; (2) dict_no_above; (3) dict_keep_n;")
    print("(4) num_topics; (5) num_passes; (6) workers")
    print("(7) processed_docs pkl file")
    print("\nSuggested Defaults: ")
    print("(1) 30; (2) 0.70; (3) 100000;")
    print("(4) 20; (5) 2; (6) 2 [or: `nproc` - 1].")
    print("(y) processed_docs.pkl.\n")
    sys.exit(2)

dict_no_below = int(sys.argv[1])
dict_no_above = float(sys.argv[2])
dict_keep_n = int(sys.argv[3])
num_topics = int(sys.argv[4])
num_passes = int(sys.argv[5])
workers = int(sys.argv[6])

processed_docs = pd.read_pickle(sys.argv[7])

# load dictionary
print("Loading data...")
dictionary = gensim.corpora.Dictionary(processed_docs)
print(f'Unfiltered dictionary contains {len(list(dictionary.values()))} features.')

# filter dictionary
Print("Filtering...")
dictionary.filter_extremes(no_below=dict_no_below, no_above=dict_no_above, keep_n=dict_keep_n)
print(f'Filtered dictionary contains {len(list(dictionary.values()))} features.')

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

# set passes and range of topic numbers to try

range_num_topics = [num_topics]

# lists to index each type of model generated below
tf_set = [[] for x in range(len(range_num_topics))]

count = 0

# build one model of each type for every `num_topics` in `range_num_topics`
# append these to `tf_set`

print("\nHyperparameters selected: ")
print(f'dict_no_below = {dict_no_below}')
print(f'dict_no_above = {dict_no_above}')
print(f'dict_keep_n = {dict_keep_n}')
print(f'num_topics = {num_topics}')
print(f'num_passes = {num_passes}')
print(f'workers = {workers}')

print("\nInitializing model training...\n")

for i in range_num_topics:
  num_topics = i
  tf_set[count] = gensim.models.LdaMulticore(corpus_tfidf, num_topics=num_topics, id2word=dictionary, passes=num_passes, workers=workers)
  
  count += 1
  
print(f'MODELS TRAINED - TF-IDF: {len(tf_set)} models; No. Passes: {num_passes}')
print(f'dict_no_below={dict_no_below}; dict_no_above={dict_no_above}; dict_keep_n={dict_keep_n}')

for x, y in zip(range_num_topics, tf_set):
  print(f'\nTF-IDF model with {x} topics: \n')
  for idx, topic in y.print_topics():
    print(f'Topic #{idx}: {topic}')

print()

# save trained model
# saves only the first trained model

filename_model = 'tf-lda.model'
filename_dict = 'tf-lda.dict'
tf_set[0].save(filename_model)
dictionary.save(filename_dict)

print("\nModel saved to current directory.")

def subprocess_cmd(command):
    process = subprocess.Popen(command,stdout=subprocess.PIPE, shell=True)
    proc_stdout = process.communicate()[0].strip()
    print(proc_stdout)

subprocess_cmd('bash tar_model.sh')

print("Model(s) backed up to ./models/saved-model")
print("DONE.")
