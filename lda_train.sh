#!/bin/bash

IMAGE_NAME=${1:-train_model}

SHARED_PATH="$HOME/lda_model"

# HYPERPARAMETERS

NO_BELOW=${2:-30}
NO_ABOVE=${3:-0.70}
KEEP_N=${4:-100000}
TOPICS=${5:-20}
PASSES=${6:-2}
WORKERS=${7:-2}
DOCS=${8:-processed_docs.pkl}

mkdir -p $SHARED_PATH;

sudo docker run -it -v $SHARED_PATH:/root $IMAGE_NAME $NO_BELOW $NO_ABOVE $KEEP_N $TOPICS $PASSES $WORKERS $DOCS;

echo "Model training complete. Models saved to $SHARED_PATH.";
