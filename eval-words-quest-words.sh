 #!/bin/bash
 
EPOCH="150"
OUTPUTDIR=gold-data
TRAINFILE=data/text8
EVALFILE=gold-data/questions-words.txt
ANSWFILE=gold-data/questions-words-words
OUTPUTFILE=gold-data/results-text8-$EPOCH"-words-quest-words.txt"
MODEL=models/text8
 
rm -f $OUTPUTFILE
python word2vec_optimized.py --use --train_data=$TRAINFILE --eval_data=$EVALFILE --answer_filename=$ANSWFILE --save_path=$MODEL --lenient &>> $OUTPUTFILE
