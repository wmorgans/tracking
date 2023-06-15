#!/bin/bash --login
#$ -cwd
#$ -pe smp.pe 2
#$ -t 1-972      # A job-array with 1800 "tasks", numbered 1...1000   



export OMP_NUM_THREADS=$NSLOTS
module load apps/binapps/anaconda3/2019.07
#
cd ..

source activate rot_2 
 
INDEX=$((SGE_TASK_ID))

params=$(sed -n "${INDEX}p" < ./generateParams/IMM3_params.csv)

saveDir="$(pwd)"

python track_with_imm_3.py $params $saveDir