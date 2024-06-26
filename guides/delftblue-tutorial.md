#### End-To-End Training & Evaluation on DelftBlue 


1. Run a delftblue job

```bash
ssh delftblue
tmux # start a sesh on the login node to keep our job alive

srun --job-name=test --nodes=1 --ntasks=1 --partition=gpu-a100 --mem=80G --cpus-per-task=32 --gpus=1 --time=06:00:00 --pty bash

tmux new -s aral 
```

Alternatively, you can simply submit a job using slurm, and then connect to it.

```bash
sbatch job.sh # included under `common`
srun --pty --overlay jobid=XXXXX bash 
```

2. Setup node

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

4. Set up python envs 

```bash
### TRAINING
cd code/common 
conda env create -n tiny 
conda activate tiny 

pip install -r requirements.txt
conda install -c conda-forge htop # for tracking CPU/mem usage
```

```bash
### EVALUATION
cd evaluation-pipeline
conda create -n babylm python=3.10
conda activate babylm 

pip install -e .[dev]
pip install wandb
pip install -U torch # m80 CUDA drivers on delftblue

unzip filter_data.zip
```

5. Run your training. For this I modified the `train_baselines.py` script with my hyperparameters. 

```bash
vim train_baselines.py # edit to do what you want
python train_baselines.py
```

Note, if your training takes really long to complete a single step (i.e. more than 5hr total), it may be due to the delftblue's slow af I/O access. You can just load the entire dataset in-memory, using the `keep_in_memory=True` in `datasets`' `load_dataset` function. 

