## How to use DelftBlue

This is not a comprehensive guide, but should help you in setting up a development environment on the cluster. To better understand how DelftBlue works, check out its [docs](https://doc.dhpc.tudelft.nl/delftblue/).

#### Setup

- **Connecting**, first, set up your ssh key (if you have not done so already) without a passphrase. In this way you'll not be asked for the password once you login.
    - You probably already have a key, check if `~/.ssh/id_rsa.pub` exists. If you need to create an ssh key: `ssh-keygen -t rsa `. 

    - Then copy the public key over to the delftblue server:
        ``` bash
        cat ~/.ssh/id_rsa.pub | ssh  <NETID>@login.delftblue.tudelft.nl 'cat >> .ssh/authorized_keys'
        ```
    
    - **Setting up**, Your home directory has only space (**5 GB**), and you may not be able to connect if it is full. So, somehow anticipate which directories will be created, create respective directories under your `/scratch` folder (**5 TB**), and symlink them to `~` These should cover most bases:
    ``` bash
    echo ${USER} # this should be your netid
    mkdir /scratch/${USER}/.local
    mkdir /scratch/${USER}/.cache
    ln -s /scratch/${USER}/.local ~/.local
    ln -s /scratch/${USER}/.cache ~/.cache
    ```

#### Running Jupyter Notebooks on DelftBlue ([documentation](https://doc.dhpc.tudelft.nl/delftblue/howtos/jupyter/))
1. After connecting to DelftBlue load the following modules:
    ``` bash
    module load 2022r2      # base libraries 
    module load openmpi     # message passing interface (for parallel computing)
    module load miniconda3  # python environment & package management
    ```
    Make sure your conda saves everything to `/scratch`, as your `/home` only has 5GB:
    ``` bash
    mkdir -p /scratch/${USER}/.conda
    ln -s /scratch/${USER}/.conda $HOME/.conda
    ```

2. Create a new conda environment and install jupyterlab:
    ``` bash
    conda create --name jupyterlab
    conda activate jupyterlab
    conda install -c conda-forge jupyterlab
    ```
    This step needs to be done only the first time, once you have created the conda environment with Jupyterlab loaded, you just need to `conda activate` the environment.

3. Create a new submission script, e.g. `nano jupyterlab.sh` (opinion: use `vim`, at least you can use the keybindings you learn in all your other programming tasks). 

    ``` bash
    #!/bin/bash
    #SBATCH --job-name=jupyter
    #SBATCH --partition=compute
    #SBATCH --time=02:00:00
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=4
    #SBATCH --mem=4G
    #SBATCH --account=Education-EEMCS-Courses-CSE3000

    # Load modules:
    module load 2022r2
    module load openmpi
    module load miniconda3

    # Set conda env:
    unset CONDA_SHLVL
    source "$(conda info --base)/etc/profile.d/conda.sh"

    conda activate jupyterlab
    cat /etc/hosts
    jupyter lab --ip=0.0.0.0 --port=8888
    conda deactivate
    ```
    From this file you can customize the resources, and the time of execution.
    
4. Submit your job:
    ``` bash
    sbatch jupyterlab.sh
    ```
    It will output a code, which is the `JOBID`.

    Check if the job is running: `squeue --me` (`ST == R`, or pending if `ST == PD`); from there you can also check the the `NODELIST` and the `JOBID`.

5. Check out the output in your `slurm-XXX.out` file:

    `cat slurm-XXX.out`

    `XXX` needs to be subtituted with the `JOBID`. You will find something along these lines:
    ```
    [I 2023-02-22 09:25:13.233 ServerApp] Jupyter Server 2.3.0 is running at:
    [I 2023-02-22 09:25:13.234 ServerApp] http://cmp047:8888/lab?token=0f42f2d94b981b3f0d972586762a72601a9477d18747f33a
    [I 2023-02-22 09:25:13.234 ServerApp]     http://127.0.0.1:8888/lab?token=0f42f2d94b981b3f0d972586762a72601a9477d18747f33a
    [I 2023-02-22 09:25:13.235 ServerApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
    [C 2023-02-22 09:25:13.336 ServerApp]

        To access the server, open this file in a browser:
            file:///home/dpalagin/.local/share/jupyter/runtime/jpserver-3094075-open.html
        Or copy and paste one of these URLs:
            http://cmp047:8888/lab?token=0f42f2d94b981b3f0d972586762a72601a9477d18747f33a
            http://127.0.0.1:8888/lab?token=0f42f2d94b981b3f0d972586762a72601a9477d18747f33a
    gio: file:///home/dpalagin/.local/share/jupyter/runtime/jpserver-3094075-open.html: No application is registered as handling this file
    ```
    Pick the second URL, which will be used to access the Jupyter lab, in this example is:
    `http://127.0.0.1:8888/lab?token=0f42f2d94b981b3f0d972586762a72601a9477d18747f33a`

    This link will not work until you do the SSH tunneling.

6. On your computer, open a new Terminal Window and create an SSH tunnel:
    ``` bash
    ssh -L 8888:cmp047:8888 <netid>@login.delftblue.tudelft.nl
    ```

    You have to replace (cmp047) with the node your job is running on i.e. with your `NODELIST`. 


7.  Open your browser and start Jupyter!

    Now that your job is running, and the tunnel to the node it is running on is open, we can start our local jupyter window. To do that, just open your internet browser, and copy the URL which you found in the `slurm-XXX.out`. 

8. Cancel a job and delete it from the queue:

    To cancel a job you need to execute this command: `scancel --name=jupyter`.
    
    Where (jupyter) must be replaced with the actual name of the job you have created.

#### Submitting BG Tasks in an Interactive Session
Say you are in an interactive shell connected to a GPU node. To run something in the background, I typically use a detached, background shell immune to holdups. In code, this translates to the following. Say you want to run a command like `python train.py`:

```sh 
nohup python train.py & disown 
```

To break that down: 

- `disown` (detach) means it is no longer a child of your current shell, so should continue running even if this shell disconnects. 
- `&` (background) means a new process is spawned for this command, meaning you can continue working in this shell. 
- `nohup` (no holdups) means that when you disconnect, the `holdup` signal sent to the command will not stop it. 

Note, DelftBlue will still kill your job (by sending a signal other than `holdup`) if the first shell you spawned is closed (e.g. because you disconnected). If you want to avoid this, just wrap that command as above. But, beware that shell will live for the duration of your session then. DelftBlue moderators do not like it if you are occupying resources without using them. 

## SSH config
To avoid having to type out your netid @ the server url every time, modify your `~/.ssh/config` to contain the following: 

```
Host delftblue
  HostName login.delftblue.tudelft.nl
  User <NETID>
```
