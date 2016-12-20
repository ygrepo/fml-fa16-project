#!/bin/bash
cd ../
for n in "$@" ; do
m=$(basename $n)
echo "Working on $m"
time python scripts/wordnet_utils.py --line --lineinfile dataold/$m --lineoutfile data/$m
time java -jar wsd/target/wsd.jar --linefile sentence-classification/data/$m --outlinefile sentence-classification/data/$m
echo "Finished with $m"
done
cd sentence-classification/
