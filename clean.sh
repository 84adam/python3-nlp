#!/bin/bash

MY_FILE=$1

cat $MY_FILE | tr -dc '[:alnum:][:space:]\n\r' | tr '[:upper:]' '[:lower:]'
