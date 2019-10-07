import bs4
import requests
from urllib.request import urlopen
from itertools import chain
from html.parser import HTMLParser
from requests import get
import os
import time
from pathlib import Path
import pandas as pd
from keras.preprocessing.text import text_to_word_sequence

# WIKI TEXT BODY SCRAPING FUNCTION

def wiki_text(url):
  response = requests.get(url)
  
  para_text = []
  
  if response is not None:
    html = bs4.BeautifulSoup(response.text, 'html.parser')
    
    title = html.select("#firstHeading")[0].text
    
    paragraphs = html.select("p")
    
    for para in paragraphs:
      para_text.append(para.text.strip())
      
  return ' '.join([x for x in para_text])

# FUNCTIONS/CLASS TO SCRAPE LINKS FROM A WEBSITE

class LinkParser(HTMLParser):
    def reset(self):
        HTMLParser.reset(self)
        self.links = iter([])

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            for name, value in attrs:
                if name == 'href':
                    self.links = chain(self.links, [value])

def gen_links(f, parser):
    encoding = f.headers.get_content_charset() or 'UTF-8'
    for line in f:
        parser.feed(line.decode(encoding))
        yield from parser.links
        
# WIKIPEDIA SPECIFIC LINK-SCRAPING FUNCTION
        
def wiki_gen_links(f, parser):
  links_list = []
  wiki_list = []
  links = gen_links(f, parser)
  for i in links:
    links_list.append(i)
  for i in links_list:
    if i[0:6] == "/wiki/":
      if ":" not in i:
        if "#" not in i:
          wiki_list.append(i[6:])
  set_links_list = [x for x in set(list(wiki_list))]
  return set_links_list

# BASE URLs FROM WHICH TO SCRAPE ADDITIONAL ARTICLES
# The set of URLs below are geared towards business, engineering, and sales

url_01 = "https://en.wikipedia.org/wiki/Graphic_design"
url_02 = "https://en.wikipedia.org/wiki/Finance"
url_03 = "https://en.wikipedia.org/wiki/Sales"
url_04 = "https://en.wikipedia.org/wiki/Customer_service"
url_05 = "https://en.wikipedia.org/wiki/Accounting"
url_06 = "https://en.wikipedia.org/wiki/Business_administration"

url_07 = "https://en.wikipedia.org/wiki/Engineering"
url_08 = "https://en.wikipedia.org/wiki/Marketing"
url_09 = "https://en.wikipedia.org/wiki/Law"
url_10 = "https://en.wikipedia.org/wiki/Business"
url_11 = "https://en.wikipedia.org/wiki/Manufacturing"
url_12 = "https://en.wikipedia.org/wiki/Value-added_reseller"

url_13 = "https://en.wikipedia.org/wiki/Technology"
url_14 = "https://en.wikipedia.org/wiki/User_experience"
url_15 = "https://en.wikipedia.org/wiki/Logic"
url_16 = "https://en.wikipedia.org/wiki/Communication"
url_17 = "https://en.wikipedia.org/wiki/Industry"
url_18 = "https://en.wikipedia.org/wiki/Electronics"

url_19 = "https://en.wikipedia.org/wiki/Energy"
url_20 = "https://en.wikipedia.org/wiki/Transport"
url_21 = "https://en.wikipedia.org/wiki/Software"
url_22 = "https://en.wikipedia.org/wiki/Mathematics"
url_23 = "https://en.wikipedia.org/wiki/System"
url_24 = "https://en.wikipedia.org/wiki/Knowledge"

base_urls = [url_01, url_02, url_03, url_04, url_05, url_06]
add_urls1 = [url_07, url_08, url_09, url_10, url_11, url_12]
add_urls2 = [url_13, url_14, url_15, url_16, url_17, url_18]
add_urls3 = [url_19, url_20, url_21, url_22, url_23, url_24]

for i in add_urls1:
  base_urls.append(i)
  
for i in add_urls2:
  base_urls.append(i)
  
for i in add_urls3:
  base_urls.append(i)

# COLLECTION OF LINKS FROM WIKIPEDIA ARTICLES

url_list = []
wiki_base = "https://en.wikipedia.org/wiki/"

for i in base_urls:
  if i not in url_list:
    url_list.append(i)
  parser = LinkParser()
  f = urlopen(i)
  wlinks = wiki_gen_links(f, parser)
  for l in wlinks:
    if l not in url_list:
      url_list.append(wiki_base + l)
      
# GATHER BODY TEXT FROM ALL ARTICLES IN url_list
# Use time.sleep(1) or greater between downloads

def wiki_all_text(url_list):
  print("Downloading {} documents...\n".format(len(url_list)))
  all_docs = []
  for i in url_list:
    print("Fetching text from: {}".format(i))
    all_docs.append(wiki_text(i))
    time.sleep(2)
  print("Download complete.\n")
  return all_docs

# RUN IT

idx = url_list
doc = wiki_all_text(url_list)

# CREATE DATAFRAME AND CSV FILE FOR EXPORT

wiki_df = pd.DataFrame({'index':[x for x in idx], 'doc':[' '.join(text_to_word_sequence(str(x))) for x in doc]})
wiki_df.to_csv('wiki_df.csv')

wiki_df.head(30)
