#!/bin/bash

python word2vec_optimized.py --min_count 1 --embedding_size 500 --epochs_to_train 150 --train_data=data/2016-12-07-text8-synsets.txt  --eval_data=data/capital-common-countries-synsets.txt --save_path=models/synsets
