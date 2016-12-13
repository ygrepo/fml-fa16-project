#!/bin/bash

#declare -a arr=("xaa")   
declare -a arr=("xaa" "xab" "xac" "xad" "xae" "xaf" "xag" "xah" "xai")   
   
DATA_DIR="pre-data/"

for f in "${arr[@]}"
do 
   if=$DATA_DIR$f
   echo "$if"
   of=$DATA_DIR$f"-l-pos.txt"
   echo "$of"
    python wordnet_utils.py --line --lineinfile $if --lineoutfile $of
done

