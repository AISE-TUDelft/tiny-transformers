#!/bin/bash -l
#
#SBATCH --job-name="blimp"
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --time=00:45:00
#SBATCH --account=Education-EEMCS-Courses-CSE3000

model_dir=$1

cd ~/CSE3000/tiny-transformers/code/evaluation-pipeline/

module load 2023r1 openmpi miniconda3 cuda/11.7

conda activate evaluation


# store model_type as decoder if model_dir contains GPT 
# else it's an encoder 
if [[ $model_dir == *"gpt"* ]]; then
    model_type="decoder"
else
    model_type="encoder"
fi

printf "\n\n\n\033[1mRunning $model_type on BLiMP for \n$model_dir\033[0m\n\n"
python babylm_eval.py $model_dir $model_type