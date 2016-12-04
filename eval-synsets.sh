#!/bin/bash

declare -a arr=("currency" "city" "family" "gram1-adj-adv" "gram2-opposite" "gram3-comparative" "gram4-superlative" 
            "gram5-present-participle" "gram6-nationality-adj" "gram7-past-tense" "gram8-plural" "gram9-plural-verbs")

outputFile="results-text8-synsets-qa-categories.txt"  
rm -f gold-data/$outputFile

for f in "${arr[@]}"
do 
   sf=$f"-synsets.txt"
   echo "$sf"
   af="text8-"$sf
   echo "Lenient=False" >> gold-data/$outputFile
   python word2vec_optimized.py --use --train_data=data/2016-12-04-text8-synsets.txt --eval_data=data/$sf --answer_filename=$af --save_path=models/synsets  &>> gold-data/$outputFile
   echo "Lenient=True" >> gold-data/$outputFile
   python word2vec_optimized.py --use --train_data=data/2016-12-04-text8-synsets.txt --eval_data=data/$sf --answer_filename=$af --save_path=models/synsets --lenient &>> gold-data/$outputFile
done
