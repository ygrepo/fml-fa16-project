#!/bin/bash

declare -a arr=("xaa")   
#declare -a arr=("xaa" "xab" "xac" "xad" "xae" "xaf" "xag" "xah" "xai")   
   
TARGET="../wsd/target/wsd.jar"
DATA_DIR="../pre-data/"
WNETFILE="../wsd/conf/extjwnl_properties.xml"

for f in "${arr[@]}"
do 
   if=$DATA_DIR"$f""-l-pos.txt"
   echo $if  
   of=$DATA_DIR"$f""-synsets.txt"
   echo $of
   java -jar ${TARGET} --wnetfile $WNETFILE --linefile $if --outlinefile $of
done
