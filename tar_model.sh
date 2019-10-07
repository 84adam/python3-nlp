#!/bin/bash

DT=$(date '+%s')

# save model
mkdir -p models/saved-model;

if [ -f "models/saved-model.tar.gz" ]; 
	then mv models/saved-model.tar.gz models/saved_model_$DT.tar.gz;
fi

mv -f ./tf-lda* models/saved-model/;

tar czf models/saved-model.tar.gz models/saved-model/;
chmod 666 models/saved-model.tar.gz;
cp models/saved-model.tar.gz $HOME/
