#!/bin/bash

#SBATCH --account=innovation
#SBATCH --job-name=sesh
#SBATCH --nodes=1
#SBATCH --partition=gpu-a100
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1

module load 2023r1 2023r1-gcc11 openmpi miniconda3 cuda/11.7 git-lfs

# start in detached mode
tmux new -s aral -d

sleep 12h

