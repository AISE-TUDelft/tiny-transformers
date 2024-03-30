## Paper Guide

> [!NOTE] 
> This document explains how to set up your paper 'the computer science way' (not sure how/why this isn't covered in academic writing). Prerequisites: have a `GitHub` and `Overleaf` account. 


In this directory, you will place your (git submodule) repository for your paper (e.g. `aral @ 72c5906`). To work locally, I suggest the [Latex-Workshop](https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop) VSCode plugin (or [TeXStudio](https://www.texstudio.org/) if you *must* download a separate IDE per language). This requires you to [install a LaTeX distribution](https://www.tug.org/texlive/) (compiler), if you have not done so already. There are other local-development alternatives, Google is your friend. 

Why make a Git repository for your paper? Firstly, the above allows you to work locally and offline. Secondly, it should help with keeping track of different versions; `git` is a version-control tool you are familiar with, after all. Thirdly, it allows you to keep a repository on your personal github account of the paper you wrote, which you deserve to show off! And lastly, switching between different sections drove me crazy when writing on Overleaf - a tabbed editor handles this much better. 

Take some time to skim the [Latex-Workshop Wiki](https://github.com/James-Yu/LaTeX-Workshop/wiki/Install) (SyncTex is quite useful). Then, have a look at [Diomidis' Latex Advice](https://github.com/dspinellis/latex-advice/tree/master), which also contains a helpful bibtex schema in case you need to manually construct a citation. 


#### Setting Up 
Brief explanation of how to set up a git project and sync it with both GitHub and Overleaf. 

1. Copy the template folder and name it appropriately. 

    ```bash 
    cp -r 'paper template' 1-hidden-layers  # copy dir and contents
    cd 1-hidden-layers                      # enter dir
    ```

2. Enter your paper folder and set up a Git repo. (I've already included a tex `.gitignore` for intermediate build files)

    ```bash 
    git init 
    git add . 
    git commit -m 'first commit' 
    ```

3. Create a new repo on your GitHub profile (can be private), and push your local repo to it ([guide](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github#adding-a-local-repository-to-github-using-git)). If you have the repository as private, make sure to add me (`Ar4l`) to it.

    ```bash 
    git remote add origin git@github.com:Ar4l/exploring-hidden-layers.git
    git push --set-upstream origin main
    ``` 

4. Set up an [Overleaf](https://www.overleaf.com/project) project, because I know Mali will ask for it. `New Project > Import from GitHub`. Note that you need to manually push/pull in the Overleaf UI. To have everything in one place, add a link to your Overleaf on the main `readme.md` as well please. 

5. Now, register your paper repo as a submodule in the main repository. Make sure to add the `-b main` flag to track the `HEAD` of the main branch. This also means you should update the `main` branch when you submit your paper for review. 

    ```bash
    cd ..  # make sure you are in the project repo, and not your paper repo
    git submodule add \
        -b main \                                           # branch to track
        git@github.com:Ar4l/exploring-hidden-layers.git \   # repo url
        1-hidden-layers                                     # submodule dir name
    ```

One caveat of `git` submodules is that, even though we tell it explicitly to track the `main` branch of your paper submodule, you will need to manually commit those changes in the communal repo. I.e., when you push changes to the main branch of your paper; remember to also update the reference in the communal repo (this will show up as `modified` when running `git status` in the communal repo)

That's all; you can leave it here. But, I leave the next two sections here if you happen to have the time. 


#### Syncing Zotero Bibliography
There exist [automatic-syncing](https://retorque.re/zotero-better-bibtex/index.html) tools which will export your bibliography right after you add a new reference. But, seeing as you only do this a few times, you can also manually export your `references.bib` from Zotero to your `tex` folder. 


#### Pushing to both GitHub and Overleaf Simultaneously
Overleaf does not automatically pull/push, you have to manually press that button under `Menu > GitHub`. Also, when working locally, it's probably convenient to push to both GitHub and Overleaf simultaneously. If you want this on your newly created Overleaf project, go to `Menu > Git` (**not** `GitHub`) and copy a url (either url should be fine). 

```bash 
# add a new remote named 'overleaf'
git remote add overleaf https://git@git.overleaf.com/6606f7b17aad8884a84247cb

# merge overleaf's newly initialised repository into your local paper repo
git pull overleaf master --no-rebase --allow-unrelated-histories

# update overleaf's repo to be aligned with your local one
git push overleaf main:master # push main branch into overleaf's master branch
```

Note that Overleaf still uses the legacy `master` branch instead of `main`, complicating push and pull a bit. It also re-initialises its own git repository (for some mysterious reason), so you need to merge it with your local one. You cannot force-push to overleaf's master either. 

```bash
git pull overleaf master # to pull into current branch 
git push overleaf main:master # push main branch into overleaf's master branch
```

Now, to finally set up pushing to both the GitHub (`origin`) and Overleaf (`overleaf`) remotes simultaneously. The easiest method is likely to create an alias for this repo under `.git/config`; but I leave this for your own exploration as I am out of time. 
