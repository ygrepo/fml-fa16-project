
EPOCH="15"
MINCUT=5
EMBSIZE=200
OUTPUTDIR=gold-data
TRAINFILE=data/2016-12-07-text8-synsets.txt
EVALFILE=gold-data/questions-words-synsets.txt
ANSWFILE=gold-data/questions-words-synsets
OUTPUTFILE=gold-data/results-text8-$EPOCH-$MINCUT-$EMBSIZE-"synsets-quest-words.txt"
MODEL=models/synsets

rm -f $OUTPUTFILE
python word2vec_optimized.py --use --train_data=$TRAINFILE --eval_data=$EVALFILE --answer_filename=$ANSWFILE --save_path=$MODEL --lenient &>> $OUTPUTFILE

tmpfile1=$(mktemp /tmp/eval-synsets.XXXXXX)
grep  -v '^Skipped' $OUTPUTFILE > $tmpfile1
tmpfile2=$(mktemp /tmp/eval-synsets.XXXXXX)
grep -v '^$' $tmpfile1 > $tmpfile2
mv $tmpfile2 $OUTPUTFILE
