#!/bin/bash

declare -a arr=("capital-common-countries")

#declare -a arr=("capital-common-countries" "capital-world" "currency" "city" "family" "gram1-adj-adv" "gram2-opposite" "gram3-comparative" "gram4-superlative" 
#            "gram5-present-participle" "gram6-nationality-adj" "gram7-past-tense" "gram8-plural" "gram9-plural-verbs")
            
TARGET="wsd/target/wsd.jar"
DATA_DIR="data/"

for f in "${arr[@]}"
do 
   if=$DATA_DIR"$f""-l-pos.txt"
   echo $if  
   of=$DATA_DIR"$f""-synsets.txt"
   echo $of
   java -jar ${TARGET} --linefile $if --outlinefile $of
done
