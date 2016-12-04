#!/bin/bash


declare -a arr=("currency" "city" "family" "gram1-adj-adv" "gram2-opposite" "gram3-comparative" "gram4-superlative" 
            "gram5-present-participle" "gram6-nationality-adj" "gram7-past-tense" "gram8-plural" "gram9-plural-verbs")

outputFile="results-text8-words-qa-categories.txt"            
for f in "${arr[@]}"
do 
   echo "$f"
   af="text8-"$f"-words"
   python word2vec_optimized.py --use --train_data=data/text8  --eval_data=data/$f.txt --answer_filename=$af --save_path=models/text8 &>> gold-data/$outputFile
done

