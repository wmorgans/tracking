#!/bin/bash --login
#$ -cwd
#$ -t 1-48

module load apps/binapps/anaconda3/2019.07

cd ..

source activate rot_2 
 
INDEX=$((SGE_TASK_ID))

params=$(sed -n "${INDEX}p" < ./generateParams/naive_rnr_params.csv)

mkdir naive_rnr

saveDir="$(pwd)"

python track_with_naive_rnr.py $params $saveDir