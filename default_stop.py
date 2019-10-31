def default_stop():
  # intersection of gensim, nltk, spacy, and sklearn stopword lists
  default = ['me', 'inc', 'shan', "needn't", 'she', '‘s', 'therefore', 'find', 'down', 'thereupon', 'without', 'up', 'yourselves', 'many', 'eleven', 'full', 'de', 're', 'wherever', 'on', 'her', 'already', 'through', 'side', 'having', 'together', 't', 'take', "'m", 'therein', 'everyone', 'himself', 'whenever', 'them', "'s", 'once', 'forty', 'only', 'must', 'hereupon', 'moreover', 'my', 'very', 'say', 'whom', 'get', 'eg', 'does', 'll', 'indeed', 'everything', 'couldnt', '’m', 'not', 'each', 'using', 'do', 've', 'cant', 'if', 'various', 'throughout', 'otherwise', 'serious', 'd', 'regarding', 'mustn', 'yourself', 'noone', 'somewhere', 'twenty', 'most', 'thick', 'describe', 'however', 'fire', 'see', 'eight', 'while', 'besides', 'neither', 'well', 'us', 'below', 'is', "won't", 'might', 'mine', 'anywhere', 'weren', "'re", "n't", 'whereupon', 'becomes', 'should', 'hereafter', 'ours', 'during', 'a', 'ltd', 'con', 'isn', 'else', 'whither', 'shouldn', 'why', 'will', 'seems', 'ie', 'every', 'someone', 'bottom', 'ain', 'needn', 'then', 'thin', 'being', 'whereafter', 'via', 'never', 'same', "haven't", 'y', 'behind', 'name', 'give', 'move', 'some', 'six', 'we', 'whole', 'than', 'myself', 'our', "wasn't", 'now', 'whether', "mustn't", 'were', 'still', 'along', 'enough', 'for', 'yours', 'whereby', 'per', 'had', 'next', 'twelve', "doesn't", 'onto', 'cry', 'seeming', 'are', 'between', 'almost', 'third', 'latter', 'by', 'nevertheless', 'in', 'across', 'though', 'kg', 'somehow', 'out', 'show', 'no', 'either', 'didn', 'computer', '’ve', 'such', 'all', 'both', 'few', "weren't", 'from', '’d', 'doing', 'alone', 'nan', 'latterly', 's', 'although', 'fifteen', 'hasn', 'own', 'due', 'whereas', 'beyond', "you'd", "shouldn't", 'whose', 'who', 'n’t', 'unless', 'something', "shan't", 'other', 'also', 'they', 'make', 'three', 'been', 'found', 'whoever', 'doesn', 'first', 'made', 'ten', 'seem', '‘ll', 'of', 'your', 'at', 'the', 'where', 'further', 'has', 'former', 'their', 'or', 'four', 'so', 'wherein', 'empty', 'among', 'mill', 'be', 'hasnt', 'used', 'go', 'amongst', 'everywhere', 'fifty', "hadn't", '’ll', 'you', 'km', 'others', 'this', 'thru', 'may', 'wouldn', 'itself', "'d", 'please', 'could', 'done', 'several', 'afterwards', 'two', 'becoming', 'those', '‘ve', 'part', 'hundred', 'system', 'upon', "wouldn't", 'meanwhile', 'thus', '’s', 'herein', 'hadn', 'put', 'toward', 'hers', 'these', 'sometime', 'don', 'nine', 'have', 'won', 'least', 'thereafter', 'often', 'nobody', 'except', 'always', '’re', "you've", 'since', 'elsewhere', 'here', 'wasn', 'as', 'less', 'there', 'one', 'anyone', 'when', 'sometimes', 'its', 'formerly', 'ca', 'thence', 'm', "don't", 'rather', 'but', 'above', 'themselves', 'his', 'haven', 'what', 'too', 'aren', 'keep', "mightn't", 'top', 'he', 'anyhow', 'co', 'around', 'etc', 'about', 'nor', 'anyway', 'hence', '‘d', 'sixty', 'mostly', 'detail', 'anything', 'bill', 'much', "she's", 'ourselves', 'fify', 'that', 'last', 'theirs', 'really', 'back', 'un', 'yet', 'just', 'was', 'an', 'ma', "isn't", "you'll", "should've", 'until', 'off', 'perhaps', 'beside', 'nowhere', 'mightn', 'sincere', "'ll", "didn't", "it's", 'am', 'again', 'even', 'which', 'front', 'can', 'within', "aren't", 'him', "you're", 'and', 'namely', 'against', '‘re', "that'll", 'with', 'whence', 'five', 'amount', 'o', 'quite', 'call', 'interest', 'none', 'before', 'fill', 'how', 'it', 'ever', 'seemed', 'i', 'because', 'thereby', 'would', '‘m', 'couldn', "couldn't", 'did', "'ve", 'under', 'after', 'more', 'become', 'nothing', 'herself', 'to', 'any', 'over', 'into', "hasn't", 'hereby', 'towards', 'amoungst', 'whatever', 'became', 'n‘t', 'beforehand', 'another', 'cannot']
  return default

if __name__ == '__main__':
  stopwords = default_stop()
  print(stopwords)