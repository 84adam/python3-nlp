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

base_urls = ['https://en.wikipedia.org/wiki/Graphic_design', 'https://en.wikipedia.org/wiki/Marketing', 
             'https://en.wikipedia.org/wiki/Communication', 'https://en.wikipedia.org/wiki/Sales', 
             'https://en.wikipedia.org/wiki/Finance', 'https://en.wikipedia.org/wiki/Accounting', 
             'https://en.wikipedia.org/wiki/Law', 'https://en.wikipedia.org/wiki/Business', 
             'https://en.wikipedia.org/wiki/Business_administration', 'https://en.wikipedia.org/wiki/Value-added_reseller', 
             'https://en.wikipedia.org/wiki/Customer_service', 'https://en.wikipedia.org/wiki/User_experience', 
             'https://en.wikipedia.org/wiki/Energy', 'https://en.wikipedia.org/wiki/Transport', 
             'https://en.wikipedia.org/wiki/Industry', 'https://en.wikipedia.org/wiki/Manufacturing', 
             'https://en.wikipedia.org/wiki/Electronics', 'https://en.wikipedia.org/wiki/Software', 
             'https://en.wikipedia.org/wiki/Engineering', 'https://en.wikipedia.org/wiki/Technology', 
             'https://en.wikipedia.org/wiki/Mathematics', 'https://en.wikipedia.org/wiki/System', 
             'https://en.wikipedia.org/wiki/Knowledge', 'https://en.wikipedia.org/wiki/Logic', 
             'https://en.wikipedia.org/wiki/Engineer', 'https://en.wikipedia.org/wiki/Microcontroller', 
             'https://en.wikipedia.org/wiki/Industrial_control_system', 'https://en.wikipedia.org/wiki/PID_controller', 
             'https://en.wikipedia.org/wiki/Control_loop', 'https://en.wikipedia.org/wiki/Programmable_logic_controller', 
             'https://en.wikipedia.org/wiki/Assembly_line', 'https://en.wikipedia.org/wiki/Robotics', 
             'https://en.wikipedia.org/wiki/Petroleum_engineering', 'https://en.wikipedia.org/wiki/Industrial_engineering', 
             'https://en.wikipedia.org/wiki/Open-source_software', 'https://en.wikipedia.org/wiki/Electrical_engineering', 
             'https://en.wikipedia.org/wiki/Computer_engineering', 'https://en.wikipedia.org/wiki/Computer_science', 
             'https://en.wikipedia.org/wiki/Mechanical_engineering', 'https://en.wikipedia.org/wiki/Microsoft_Windows', 
             'https://en.wikipedia.org/wiki/Operating_system', 'https://en.wikipedia.org/wiki/Computer_program', 
             'https://en.wikipedia.org/wiki/Human%E2%80%93computer_interaction', 'https://en.wikipedia.org/wiki/History', 
             'https://en.wikipedia.org/wiki/Art', 'https://en.wikipedia.org/wiki/Music', 'https://en.wikipedia.org/wiki/Food', 
             'https://en.wikipedia.org/wiki/Education', 'https://en.wikipedia.org/wiki/Health', 
             'https://en.wikipedia.org/wiki/Medicine', 'https://en.wikipedia.org/wiki/Politics', 
             'https://en.wikipedia.org/wiki/Management', 'https://en.wikipedia.org/wiki/Chemistry', 
             'https://en.wikipedia.org/wiki/Biology', 'https://en.wikipedia.org/wiki/Physics', 
             'https://en.wikipedia.org/wiki/Geology', 'https://en.wikipedia.org/wiki/Astronomy', 
             'https://en.wikipedia.org/wiki/Anthropology', 'https://en.wikipedia.org/wiki/Sociology', 
             'https://en.wikipedia.org/wiki/Psychology', 'https://en.wikipedia.org/wiki/Science', 
             'https://en.wikipedia.org/wiki/Formal_science', 'https://en.wikipedia.org/wiki/Natural_science', 
             'https://en.wikipedia.org/wiki/Social_science', 'https://en.wikipedia.org/wiki/Game_theory', 
             'https://en.wikipedia.org/wiki/Network_theory', 'https://en.wikipedia.org/wiki/Artificial_neural_network', 
             'https://en.wikipedia.org/wiki/Broadcast_network', 'https://en.wikipedia.org/wiki/Electrical_network', 
             'https://en.wikipedia.org/wiki/Social_networking_service', 
             'https://en.wikipedia.org/wiki/Telecommunications_network', 'https://en.wikipedia.org/wiki/Computer_network', 
             'https://en.wikipedia.org/wiki/Transport_network', 'https://en.wikipedia.org/wiki/Money', 
             'https://en.wikipedia.org/wiki/Bitcoin', 'https://en.wikipedia.org/wiki/Gold', 
             'https://en.wikipedia.org/wiki/Silver', 'https://en.wikipedia.org/wiki/Fiat_money', 
             'https://en.wikipedia.org/wiki/Bank', 'https://en.wikipedia.org/wiki/Economics', 
             'https://en.wikipedia.org/wiki/Production_(economics)', 'https://en.wikipedia.org/wiki/Service_(economics)', 
             'https://en.wikipedia.org/wiki/Utility', 'https://en.wikipedia.org/wiki/The_arts', 
             'https://en.wikipedia.org/wiki/Philosophy', 'https://en.wikipedia.org/wiki/Theatre', 
             'https://en.wikipedia.org/wiki/Film', 'https://en.wikipedia.org/wiki/Dance', 
             'https://en.wikipedia.org/wiki/Fine_art', 'https://en.wikipedia.org/wiki/Applied_arts', 
             'https://en.wikipedia.org/wiki/Linguistics', 'https://en.wikipedia.org/wiki/Slang', 
             'https://en.wikipedia.org/wiki/Sarcasm', 'https://en.wikipedia.org/wiki/Culture', 
             'https://en.wikipedia.org/wiki/Security', 'https://en.wikipedia.org/wiki/Media', 
             'https://en.wikipedia.org/wiki/List_of_countries_by_spoken_languages', 'https://en.wikipedia.org/wiki/Humanities', 
             'https://en.wikipedia.org/wiki/Sport', 'https://en.wikipedia.org/wiki/Relationship', 
             'https://en.wikipedia.org/wiki/Religion', 'https://en.wikipedia.org/wiki/Faith', 
             'https://en.wikipedia.org/wiki/Spirituality', 'https://en.wikipedia.org/wiki/Literature', 
             'https://en.wikipedia.org/wiki/Fiction', 'https://en.wikipedia.org/wiki/Nonfiction', 
             'https://en.wikipedia.org/wiki/Classics', 'https://en.wikipedia.org/wiki/Western_world', 
             'https://en.wikipedia.org/wiki/Eastern_world', 'https://en.wikipedia.org/wiki/Renaissance', 
             'https://en.wikipedia.org/wiki/History_by_period', 'https://en.wikipedia.org/wiki/List_of_time_periods', 
             'https://en.wikipedia.org/wiki/Category:History_of_science_and_technology_by_country']

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
    time.sleep(0.5)
  print("Download complete.\n")
  return all_docs

# RUN IT

idx = url_list
doc = wiki_all_text(url_list)

# CREATE DATAFRAME AND CSV FILE FOR EXPORT

wiki_df = pd.DataFrame({'index':[x for x in idx], 'doc':[' '.join(text_to_word_sequence(str(x))) for x in doc]})
wiki_df.to_csv('wiki_df.csv')

wiki_df.head(30)
