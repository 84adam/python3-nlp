# Latent Dirichlet Allocation-based Topic Modeling using Gensim

SEE: https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation

**REQUIRED: Python 3.6+**

### Text Preprocessing

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
  - As an example, you can use a collection of Wikipedia articles. Download an example set [here](https://www.dropbox.com/s/45s8t1y0ixxe4zh/wiki_df.csv?dl=0) (10,000+ articles).
  - Use **`get_wiki.py`** to download your own set of articles from Wikipedia, changing the list of top-level URLs from which to scrape articles (and those of articles linked within each of these documents). NOTE: Choosing ~24 generic topics, and recursing down one level to all linked articles, results in ~10,000 articles being downloaded.
  
### LDA Model Training
  
- `train_model.py` accepts a pickle-formatted file representing the preprocessed text
  - If you wish you can use the preprocessed `.pkl` file produced from the sample `wiki_df.csv` file linked above. Download this proprocessed set of Wikipedia articles [here](https://www.dropbox.com/s/picanyvasfrc91g/processed_docs.pkl?dl=0).
  - This is loaded into memory and a dictionary is created using Gensim
  - Subsequently, Topic Model training is initialized
  - Hyperparameters can be defined by passing command-line arguments to `train_model.py`, for example:
    - `dict_no_below = 30` - no words appearing less than {n} times in the corpus
    - `dict_no_above = 0.70` - no words appearing in more than {n}% of all documents in the corpus
    - `dict_keep_n = 100000` - dictionary to be comprised of a maximum of {n} features
    - `num_topics = 20` - create a model with {n} topics
    - `num_passes = 2` - perform {n} passes through the corpus during model training
    - `workers = 2` - use {n} CPU cores during model training
      - Recommended formula for `workers`: `$ echo "$(nproc) - 1" | bc`
      - Using 2 or more CPU cores allows you to take advantage of LdaMulticore in Gensim.
    - `processed_docs = pd.read_pickle('processed_docs.pkl')` - provide the name of a pickle-formatted file containing a Pandas Series object with preprocessed text
   - The output of `train_model.py` is a list of topics that have been derived from the corpus of preprocessed documents.
   - The Gensim model files, state, and dictionary are saved to the current directory, as well as backed up to `./models/saved-model/`, using the accompanying bash script: `tar_model.sh`.
   
Example of Topics Derived:

```
Topic #0: 0.005*"tourism" + 0.003*"travel" + 0.002*"tourist" + 0.002*"gnd" + 0.002*"aircraft" + 0.002*"tour" + 0.001*"industri" + 0.001*"citi" + 0.001*"roman" + 0.001*"flight"
Topic #1: 0.003*"engin" + 0.003*"manag" + 0.002*"transport" + 0.002*"social" + 0.002*"softwar" + 0.002*"econom" + 0.002*"busi" + 0.002*"system" + 0.002*"design" + 0.002*"tourism"
Topic #2: 0.004*"game" + 0.003*"bu" + 0.003*"water" + 0.002*"servic" + 0.002*"media" + 0.002*"fare" + 0.002*"digit" + 0.002*"compani" + 0.002*"travel" + 0.002*"hotel"
Topic #3: 0.004*"mathemat" + 0.003*"philosophi" + 0.003*"theori" + 0.003*"displaystyl" + 0.003*"logic" + 0.003*"philosoph" + 0.002*"scienc" + 0.002*"knowledg" + 0.002*"softwar" + 0.002*"comput"
Topic #4: 0.005*"engin" + 0.002*"univers" + 0.002*"scienc" + 0.002*"system" + 0.002*"professor" + 0.002*"award" + 0.002*"electr" + 0.002*"transistor" + 0.002*"mathemat" + 0.002*"robot"
Topic #5: 0.003*"parser" + 0.002*"milki" + 0.002*"award" + 0.002*"galaxi" + 0.002*"output" + 0.002*"locomot" + 0.001*"svg" + 0.001*"ubuntu" + 0.001*"stone" + 0.001*"design"
Topic #6: 0.003*"heat" + 0.003*"thermodynam" + 0.003*"energi" + 0.002*"sale" + 0.002*"compani" + 0.001*"sector" + 0.001*"tourism" + 0.001*"market" + 0.001*"busi" + 0.001*"steam"
Topic #7: 0.004*"energi" + 0.003*"electr" + 0.003*"solar" + 0.003*"wind" + 0.002*"confucian" + 0.002*"ga" + 0.002*"law" + 0.002*"turbin" + 0.002*"power" + 0.002*"magnet"
Topic #8: 0.004*"compani" + 0.003*"app" + 0.003*"corpor" + 0.002*"sharehold" + 0.002*"jstor" + 0.002*"oil" + 0.002*"busi" + 0.002*"director" + 0.002*"store" + 0.001*"commun"
Topic #9: 0.004*"isbn" + 0.003*"brahman" + 0.002*"hindu" + 0.002*"yoga" + 0.002*"vedanta" + 0.002*"philosophi" + 0.002*"hinduism" + 0.002*"digit" + 0.002*"buddhism" + 0.002*"jain"
```

### Infer Topics of New, Unseen Document

**`infer_topics.py`**:

- requires a text string as input
- requires that saved gensim model and dictionaries be present in the same directory
  - `tf-lda.model` and `tf-lda.dict`
  
### Compare the Topic Probability Distributions of Any Two Documents

**`jsd_topics.py`**:

- requires two text strings as input (ex. use `"$(cat doc1.txt)" "$(cat doc2.txt)"`)
  - these are `sys.argv[2]` and `sys.argv[3]` (second and third arguments)
- requires the Gensim LDA model files and stopword files be available on the local filesystem
  - Provide the PATH for: `tf-lda.model`, `tf-lda.dict`, `stop.txt`, `unstop.txt` as `sys.argv[1]` (first argument)
  - `stop.txt` and `unstop.txt` can be empty, but must exist in the PATH specified
- returns a Jensen-Shannon Distance score indicating how similar or different the two documents are from each other
  - a lower score (closer to 0) means the two documents are similar
  - a higher score (closer to 1) means the documents are different
