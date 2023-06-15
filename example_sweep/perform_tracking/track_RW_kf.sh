#!/bin/bash --login
#$ -cwd
#$ -t 1-360       # A job-array with 1800 "tasks", numbered 1...1000   



export OMP_NUM_THREADS=$NSLOTS
module load apps/binapps/anaconda3/2019.07
#
cd ..

source activate rot_2 
 
INDEX=$((SGE_TASK_ID))

params=$(sed -n "${INDEX}p" < ./generateParams/RW_KF_params.csv)

mkdir naive_RW

saveDir="$(pwd)"

python track_with_kf_rw.py $params $saveDir