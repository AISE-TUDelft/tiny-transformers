
1. Run a delftblue job 

```bash
ssh delftblue

srun --job-name=test --nodes=1 --ntasks=1 --partition=gpu-a100 --mem=80G --cpus-per-task=32 --gpus=1 --time=06:00:00 --pty bash

tmux new -s aral 
```

2. Setup env 

```bash
module load 2023r1 2023r1-gcc11 openmpi miniconda3 cuda/11.7 git-lfs

git clone https://github.com/Ar4l/dotfiles
./dotfiles/setup.sh

conda init # will throw errors but does what we want
exec bash 
```

3. Clone repository

```bash
# don't work in your home directory
cd /scratch/addemoor

# if git cloning directly, make sure you have a ssh key 
# on delftblue linked to your github account. 
git clone git@github.com:AISE-TUDelft/tiny-transformers
cd tiny-transformers
git submodule init
git submodule update --init --recursive

# alternatively, you can move it from your device to delftblue 
# with either scp or rsync 
scp -r . delftblue:/scratch/addemoor/tiny-transformers-2
rsync . delftblue:/scratch/addemoor/tiny-transformers-3
```

4. Set up envs 

```bash
cd tiny-transformers/code/common 

### TRAINING
conda env create -n tiny 
pip install -r requirements.txt 
# also adding htop in there for convenience 
conda install -c conda-forge htop
```

```bash
### EVALUATION
cd evaluation-pipeline
conda env create -n babylm 
conda activate babylm 

pip install -r .[dev]
pip install wandb
```

5. Run training & evaluation. For this I modified the `train_baselines.py` script with my hyperparameters. 
