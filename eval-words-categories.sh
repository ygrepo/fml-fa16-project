#!/bin/bash


#declare -a arr=("capital-common-countries" "capital-world")
declare -a arr=("capital-common-countries" "capital-world" "currency" "city" "family" "gram1-adj-adv" "gram2-opposite" "gram3-comparative" "gram4-superlative" 
            "gram5-present-participle" "gram6-nationality-adj" "gram7-past-tense" "gram8-plural" "gram9-plural-verbs")

EPOCH="150"
TRAINFILE=data/text8
OUTPUTFILE=gold-data/results-text8-$EPOCH"-words-categories.txt"    
MODEL=models/text8
 
rm -f $OUTPUTFILE
for f in "${arr[@]}"
do 
    EVALFILE=gold-data/$f.txt
    echo $EVALFILE
    ANSWFILE=gold-data/"text8-"$f"-words"
    python word2vec_optimized.py --use --train_data=$TRAINFILE  --eval_data=$EVALFILE --answer_filename=$ANSWFILE --save_path=$MODEL &>> $OUTPUTFILE
done
tmpfile1=$(mktemp /tmp/eval-words.XXXXXX)
grep  -v '^Skipped' $OUTPUTFILE > $tmpfile1
tmpfile2=$(mktemp /tmp/eval-words.XXXXXX)
grep -v '^$' $tmpfile1 > $tmpfile2
mv $tmpfile2 $OUTPUTFILE