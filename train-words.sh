#!/bin/bash

MINCOUNT=5
EMBSIZE=200
EPOCHS=300
TRAININGDATA=data/text8
EVALDATA=gold-data/capital-common-countries.txt
MODEL=models/text8

python word2vec_optimized.py --min_count $MINCOUNT --embedding_size $EMBSIZE --epochs_to_train $EPOCHS --train_data=$TRAININGDATA  --eval_data=$EVALDATA --save_path=$MODEL
