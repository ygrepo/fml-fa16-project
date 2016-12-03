#!/bin/bash

python word2vec_optimized.py --use --interactive --train_data=data/text8-synsets.txt  --eval_data=data/questions-answers-synsets.txt --save_path=models/synsets
