#!/bin/bash

STREAMINFILE="../data/text8.zip"
STREAMOUTFILE="../data/text8-l-pos.txt"
python ../syn2vec/wordnet_utils.py --stream --zip --streaminfile $STREAMINFILE --streamoutfile $STREAMOUTFILE
