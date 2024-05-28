#!/bin/sh
#
#SBATCH --job-name="pretrain"
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --time=00:05:00 # debug mode
#SBATCH --account=Education-EEMCS-Courses-CSE3000

module load 2023r1 openmpi miniconda3 cuda/11.7

cd ~/CSE3000/tiny-transformers/code/eugene/
srun ./pretrain_all.sh 0 true # debug mode