#!/bin/bash

DT=$(date '+%s')

# save model
mkdir -p saved-model-$DT;
mkdir -p $HOME/lda_model;
cp ./tf-lda* saved-model-$DT/;
tar czf saved-model-$DT.tar.gz saved-model-$DT/;
cp saved-model-$DT.tar.gz $HOME/lda_model/
