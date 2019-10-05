# Latent Dirichlet Allocation-based Topic Modeling using Gensim

SEE: https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation

REQUIRED: Python 3.6+

- `preprocess.py` accepts a local CSV file as input
  - The CSV file can contain any number of documents or strings of arbitary length in a given column
  - `preprocess.py` will automatically locate the column with the largest amount of text in terms of average words per cell in a given column
  - The 'largest' text column from the CSV is converted to a Pandas Series object and then passed through text preprocessing steps, including:
    - stopword removal, using any number of custom stopwords, as well as the set of stopwords from:
      - `sklearn`
      - `nltk`
      - `spacy`
      - `gensim`
    - lemmatization and stemming
    - tokenization
  - The result of the preprocessing step is a pickled Pandas Series object: 'processed_docs.pkl'
  
- `ppt_model.py` accepts a pickle-formatted file representing the preprocessed text
  - This is loaded into memory and a dictionary is created using Gensim
  - Subsequently, Topic Model training is initialized
  - Hyperparameters can be defined by passing command-line arguments to `ppy_model.py`
  - Alternatively, if using Docker **[WIP]**, the script `lda_train.sh` can be used to pass default arguments:
    - `lda_train.sh` contains a set of recommended hyperparameters for model-training
    - These are passed from the bash script as arguments to the Docker image:
      - `dict_no_below = 20` - no words appearing less than {n} times in the corpus
      - `dict_no_above = 0.85` - no words appearing in more than {n}% of all documents in the corpus
      - `dict_keep_n = 100000` - dictionary to be comprised of a maximum of {n} features
      - `num_topics = 10` - create a model with {n} topics
      - `num_passes = 2` - perform {n} passes through the corpus during model training
      - `workers = 2` - use {n} CPU cores during model training - recommended formula: `$ echo "$(nproc) - 1" | bc` to take advantage of LdaMulticore in Gensim
      - `processed_docs = pd.read_pickle('processed_docs.pkl')` - provide the name of a pickle-formatted file containing a Pandas Series object with preprocessed text
   - The output of the `ppt_model.py` program is a list of topics that have been derived from the corpus of preprocessed documents, as well as an archive containing all of the Gensim model, dictionary, and state files.
   
Example of Topics Derived:

<WIP>
  
<WIP: Inference program, using the saved model>
