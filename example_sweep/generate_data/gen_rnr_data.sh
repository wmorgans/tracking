#!/bin/bash --login
#$ -cwd
#$ -t 1-18

module load apps/binapps/anaconda3/2019.07

source activate rot_2 
 
INDEX=$((SGE_TASK_ID))


params=$(sed -n "${INDEX}p" < rnr_params.csv)

mkdir rnr_data

saveDir="$(pwd)/rnr_data/"

python gen_rnr.py $params $saveDir



