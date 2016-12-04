#!/bin/bash

#python word2vec_optimized.py --use --train_data=data/text8-synsets.txt  --eval_data=data/questions-answers-synsets.txt --save_path=models/synsets
python word2vec_optimized.py --use --train_data=data/text8-synsets.txt  --eval_data=data/capital-common-countries-synsets.txt --save_path=models/synsets
