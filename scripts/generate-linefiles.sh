#!/bin/bash

declare -a arr=("questions-words")
#declare -a arr=("capital-common-countries" "capital-world" "currency" "city" "family" "gram1-adj-adv" "gram2-opposite" "gram3-comparative" "gram4-superlative" 
#            "gram5-present-participle" "gram6-nationality-adj" "gram7-past-tense" "gram8-plural" "gram9-plural-verbs")

DATA_DIR="../gold-data/"

for f in "${arr[@]}"
do 
   if=$DATA_DIR$f".txt"
   echo "$if"
   of=$DATA_DIR$f"-l-pos.txt"
   echo "$of"
    python ../syn2vec/wordnet_utils.py --line --lineinfile $if --lineoutfile $of
done

