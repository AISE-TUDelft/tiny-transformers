#!/bin/bash -l
#
#SBATCH --job-name="superglue"
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --account=Education-EEMCS-Courses-CSE3000

model_dir=$1

cd ~/CSE3000/tiny-transformers/code/evaluation-pipeline/

module load 2023r1 openmpi miniconda3 cuda/11.7

conda activate evaluation

printf "\n\n\n\033[1mRunning SuperGLUE for \n$model_dir\033[0m\n\n"
bash finetune_all_tasks.sh $model_dir