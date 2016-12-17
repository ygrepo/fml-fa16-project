#!/bin/bash

STREAMINFILE="data/wiki5k.txt"
STREAMOUTFILE="data/wiki5k-l-pos.txt"
python wordnet_utils.py --stream --streaminfile $STREAMINFILE --streamoutfile $STREAMOUTFILE
