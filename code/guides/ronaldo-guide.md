## Ronaldo Guide

Ronaldo is a shared server among the SERG (Software Engineering Research Group). While it is often available, you need to be careful to not `rm -rf /` or do anything crazy that may affect the others relying on this server. 

> This guide explains my `ronaldo` workflow. Specifically: 
> 1. How to connect via `ssh`. 
> 2. Managing sessions using `tmux`.
> 3. Working in `docker` containers. 
> 4. Connecting VSCode to a dev container on `ronaldo`. 

Ronaldo has the following hardware:
- 2x NVidia 3080 GPU (please don't use both at once, 5+ people need to share them).
  See specs & usage with `nvidia-smi`
- 128 CPU cores (please don't use all at once; start with like `20`).
  See specs with `lscpu`.

#### Connecting
I've set up my local `~/.ssh/config` as follows to connect.

```
Host ronaldo
  HostName ronaldo.ewi.tudelft.nl
  User ademoor
```

Then connecting is as simple as: `ssh ronaldo`

#### Managing Sessions 
Before we get to development, I would recommend spending 5 minutes to learn about `tmux`. This spins up a 'bash server', which will manage your sessions and keep stuff running if you disconnect. 

```bash
tmux new -s aral     # create new tmux server with name `aral`
tmux ls              # list running tmux servers
tmux attach -t aral  # enter the `aral` server
```

`tmux` is a tool for multiplexing and managing bash sessions as panes. `C-b` means press `Ctrl` and `b` at the same time, which is the command prefix in `tmux`. 
- `C-b %` for vsplit; `C-b "` for hsplit.
- `C-b <arrow>` for navigating panes.
- `C-b x` to kill a pane. 
- `C-b c` for new window; `n` and `p` to navigate to next/previous. 
- `C-b ,` to rename window.
- `C-b d` to detach session, closing `tmux` (keeps running in background)
- `tmux ls` see running sessions. 
- `tmux attach -t 0` to attach to session.
- `C-b z` fullscreen
- `C-b C-<arrow>` resize pane (on MacOS, you need to press `esc-<arrow>`)

#### Quality-Of-Life
**(optional, but highly recommend)** People always praise linux' customisability, but I think that's mainly because the defaults are absolutely awful. If you'd like, you can use my dotfiles as a starting point. 

```bash
# in your ~ directory
git clone https://github.com/Ar4l/dotfiles 
./dotfiles/setup.sh
```

This will add 
- add conda env and git branch to your prompt
- A new command `nv`, to watch the GPU usage instead of being single-use (highlights changes in GPU usage by polling every 0.5s). 
- allow you to navigate `tmux` with `C-b <hjkl>`, `vim`-style
- a bunch of other things that you'd intuitively expect to work, but don't (e.g. scrolling in `tmux` windows without pressing `C-b [` every time). 
- Selecting text in a tmux window automatically copies it to your (local) clipboard. 
- random errors (but non-breaking afaik) because I just created this repo, PRs are welcome!

The new stuff requires you to restart your tmux server, so do that real quick and get used to some of the commands you'll use often. Additionally, the changes may not be present in your first `bash` shell, but just run `bash` again. Why? I don't want to know – I leave you with this lovely diagram. 

![](https://blog.flowblok.id.au/static/images/shell-startup.png)

#### Containerising
People often use `docker` as a development container. This makes it easy to develop while being certain you are not affecting anyone else on the server, as long as you set up your container correctly – I.e. mount only your home directory in the container, not the entire server. I use a minimal `jupyter` container from `quay.io`:

```sh
# on ronaldo, at /home/aral
docker run \
	--name aral \          # name for the container
	--user root \          # user in the container (root, i.e. su)
	--rm \                 # deletes container after you shut it down
	--gpus '"device=0"' \  # which gpus to use (set to 'all' for 2 GPUs)
	--shm-size=20g \
	-d \                   # run container in background (detach)

	# mount cwd into container's /home/jovyan/work
	-v "${PWD}":/home/jovyan/work \

	# container to use
	quay.io/jupyter/minimal-notebook
```

I've annotated each flag above, but it may be easier to just copy the following equivalent command:

```sh
# on ronaldo, at /home/aral
docker run --name <YOUR_NAME> --user root --rm --gpus '"device=0"' --shm-size=20g -d -v "${PWD}":/home/jovyan/work quay.io/jupyter/minimal-notebook
```

The `quay.io/jupyter/minimal-notebook` runs a jupyter server in the container. But, unlike on `delftblue`, you can actually connect to it with your IDE (explained below). Also, the `jovyan` name is presumably the person who created the container, don't worry about it – all you should know is your `pwd` is mounted in their `~/work` directory, in the container. 

For now, some more handy `docker` commands:

- Currently running containers with `docker stats`. 
- Stop a container with `docker kill <name>`, or gracefully shut-down with `docker stop <name>`.
- Enter container with `docker exec -it <name> bash`. I.e. execute `bash` in the container `<name>` in `-it` interactive mode. 

#### Getting to Work
Okay, now assuming you have the above `docker` dev container running on `ronaldo`, you can connect to it from VSCode as follows:

- Install `Docker Dev Containers` and `Remote Development (ssh)` from the Extension Marketplace.
- In the command palette, select `Connect Current Window to Host`, and select `ronaldo` (if not in your `~/.ssh/config`, you have to type out its address)
- Once connected, open the command palette again; select `Attach to Dev Container`, and connect to your container. 

That's it! You should now be able to work on `ronaldo`, via VSCode, as if it is your local computer. 
