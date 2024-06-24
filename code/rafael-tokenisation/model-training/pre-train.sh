#!/bin/bash
#SBATCH --job-name=10k-bpe_gpt_and_bert
#SBATCH --partition=gpu-a100
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --account=Education-EEMCS-Courses-CSE3000

# Load modules:
module load 2022r2
module load openmpi
module load miniconda3

# Set conda env:
conda activate jupyterlab
cat /etc/hosts
jupyter lab --ip=0.0.0.0 --port=8888
conda install --file requirements.txt
srun python pre_train_script.py > 10k-bpe_gpt_and_bert.log 2>&1
conda deactivate