#!/bin/bash -l
#
#SBATCH --job-name="pretrain"
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --time=05:00:00
#SBATCH --account=Education-EEMCS-Courses-CSE3000

cd ~/CSE3000/tiny-transformers/code/eugene/

module load 2023r1 openmpi miniconda3 cuda/11.7

conda activate pretrain

# Assign arguments to variables
DEBUG=$1
MODEL_TYPE=$2
SPARSITY_TYPE=$3
SPARSITY_LEVEL=$4

if [ "$DEBUG" = "true" ]; then
    srun python pretrain.py --debug --model_type $MODEL_TYPE --gpu 0 --sparsity_type $SPARSITY_TYPE --sparsity_level $SPARSITY_LEVEL
else
    srun python pretrain.py --model_type $MODEL_TYPE --gpu 0 --sparsity_type $SPARSITY_TYPE --sparsity_level $SPARSITY_LEVEL
fi
