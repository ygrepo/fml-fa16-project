#!/bin/bash

STREAMINFILE="data/text8.zip"
STREAMOUTFILE="data/text8-l-pos.txt"
python wordnet_utils.py --stream --zip --streaminfile $STREAMINFILE --streamoutfile $STREAMOUTFILE