 #!/bin/bash
 
EPOCH="150"
OUTPUTDIR=../gold-data
TRAINFILE=../data/text8
EVALFILE=../gold-data/questions-words.txt
ANSWFILE=../gold-data/questions-words-words
OUTPUTFILE=../gold-data/results-text8-$EPOCH"-words-quest-words.txt"
MODEL=../models/text8
 
rm -f $OUTPUTFILE
python ../syn2vec/word2vec_optimized.py --use --train_data=$TRAINFILE --eval_data=$EVALFILE --answer_filename=$ANSWFILE --save_path=$MODEL --lenient &>> $OUTPUTFILE


tmpfile1=$(mktemp /tmp/eval-words.XXXXXX)
grep  -v '^Skipped' $OUTPUTFILE > $tmpfile1
tmpfile2=$(mktemp /tmp/eval-words.XXXXXX)
grep -v '^$' $tmpfile1 > $tmpfile2
mv $tmpfile2 $OUTPUTFILE
