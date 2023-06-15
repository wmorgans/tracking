#!/bin/bash --login
#$ -cwd
#$ -t 1-15

module load apps/binapps/anaconda3/2019.07

source activate rot_2 
 
INDEX=$((SGE_TASK_ID))

params=$(sed -n "${INDEX}p" < RW_params.csv)

mkdir RW_data

saveDir="$(pwd)/RW_data/"

python genRW.py $params $saveDir



