#!/bin/bash
#PBS -l nodes=1:ppn=1:gpus=1,walltime=2:30:00,mem=16GB
#PBS -N fomlProEval
#PBS -M ue225@nyu.edu
#PBS -m abe
#PBS -e localhost:/scratch/ue225/${PBS_JOBNAME}.e${PBS_JOBID}
#PBS -o localhost:/scratch/ue225/${PBS_JOBNAME}.o${PBS_JOBID}

OUT_FOLDER=$HOME/fomltestout
LOG_FOLDER=$SCRATCH/foml/
d=SST1

cd $PBS_JOBTMP
cp -r $HOME/FML-FA16-Project ./
cd FML-FA16-Project/sentence-classification
module load torch/gnu/20160623 

time  th main.lua -data $d.hdf5 -cudnn 1 -gpuid 1

zip -r $PBS_JOBID.zip results
curl --upload-file $PBS_JOBID.zip https://transfer.sh/$PBS_JOBID.zip > $OUT_FOLDER/$PBS_JOBID
cp $PBS_JOBID.zip $OUT_FOLDER/
mv $SCRATCH/${PBS_JOBNAME}.e${PBS_JOBID} $LOG_FOLDER
mv $SCRATCH/${PBS_JOBNAME}.o${PBS_JOBID} $LOG_FOLDER
exit 0;


