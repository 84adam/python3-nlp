#!/bin/bash

DT=$(date '+%s')

# save model
mkdir -p saved-model-$DT;
cp ./tf-lda* saved-model-$DT/;
tar czf saved-model-$DT.tar.gz saved-model-$DT/;
chmod 666 saved-model-$DT.tar.gz;
cp saved-model-$DT.tar.gz $HOME/
