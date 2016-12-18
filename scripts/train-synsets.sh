#!/bin/bash

MINCOUNT=5
EMBSIZE=200
EPOCHS=150
TRAININGDATA=../data/2016-12-07-text8-synsets.txt
EVALDATA=../gold-data/capital-common-countries.txt
MODEL=../models/synsets

python ../syn2vec/word2vec_optimized.py --min_count $MINCOUNT --embedding_size $EMBSIZE --epochs_to_train $EPOCHS --train_data=$TRAININGDATA  --eval_data=$EVALDATA --save_path=$MODEL
