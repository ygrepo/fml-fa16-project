#!/bin/bash

python word2vec_optimized.py --epochs_to_train 150 --train_data=data/2016-12-04-text8-synsets.txt  --eval_data=data/capital-common-countries-synsets.txt --save_path=models/synsets
