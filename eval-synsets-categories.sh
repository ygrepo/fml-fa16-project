#!/bin/bash

declare -a arr=("capital-common-countries")
#declare -a arr=("capital-common-countries" "capital-world" "currency" "city" "family" "gram1-adj-adv" "gram2-opposite" "gram3-comparative" "gram4-superlative" 
#            "gram5-present-participle" "gram6-nationality-adj" "gram7-past-tense" "gram8-plural" "gram9-plural-verbs")


EPOCH="150"
MINCUT=5
EMBSIZE=200
TRAINFILE=data/2016-12-07-text8-synsets.txt
OUTPUTFILE=gold-data/results-text8-$EPOCH-$MINCUT-$EMBSIZE-"-synsets-categories.txt"    
MODEL=models/synsets
rm -f $OUTPUTDIR/$outputFile

for f in "${arr[@]}"
do 
   echo "Lenient=True" >> gold-data/$outputFile
   python word2vec_optimized.py --use --train_data=$TRAINFILE --eval_data=data/$sf --answer_filename=$af --save_path=models/synsets --lenient &>> $OUTPUTDIR/$outputFile
   
   EVALFILE=gold-data/$f"-synsets.txt"
   echo $EVALFILE
   ANSWFILE=gold-data/"text8-"$f"-synsets"
   python word2vec_optimized.py --use --train_data=$TRAINFILE  --eval_data=$EVALFILE --answer_filename=$ANSWFILE --save_path=$MODEL --lenient &>> $OUTPUTFILE
done
tmpfile1=$(mktemp /tmp/eval-synsets.XXXXXX)
grep  -v '^Skipped' $OUTPUTFILE > $tmpfile1
tmpfile2=$(mktemp /tmp/eval-synsets.XXXXXX)
grep -v '^$' $tmpfile1 > $tmpfile2
mv $tmpfile2 $OUTPUTFILE

