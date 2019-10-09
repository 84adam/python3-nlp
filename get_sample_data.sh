#!/bin/bash

wget -O wiki_df.csv https://www.dropbox.com/s/45s8t1y0ixxe4zh/wiki_df.csv?dl=0 ;
wget -O processed_docs.pkl https://www.dropbox.com/s/picanyvasfrc91g/processed_docs.pkl?dl=0 ;
wget -O new_wiki_docs.tar.gz https://www.dropbox.com/s/cp5kj3ld1382kk9/new_wiki_docs.tar.gz?dl=0 ;

tar xzf new_wiki_docs.tar.gz ;

echo "Downloaded 'wiki_df.csv', 'processed_docs.pkl', and 'new_wiki_docs'." ;
echo "new_wiki_docs/ contains: " ;
ls new_wiki_docs
