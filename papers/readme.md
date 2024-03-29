## Paper Guide

> [!NOTE] This document explains how to set up your paper 'the computer science way' (not sure how/why this isn't covered in academic writing). 


In this directory, you will place your (git submodule) repository for your paper. To work locally, I suggest the [Latex-Workshop](https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop) VSCode plugin (or [TeXStudio](https://www.texstudio.org/) if you *must* download a separate IDE per language). This requires you to [install a LaTeX distribution](https://www.tug.org/texlive/) (compiler), if you have not done so already. There are other local-development alternatives, Google is your friend. 

Why make a Git repository for your paper? Firstly, the above allows you to work locally and offline. Secondly, it should help with keeping track of different versions; `git` is a version-control tool you are familiar with, after all. Thirdly, it allows you to keep a repository on your personal github account of the paper you wrote, which you deserve to show off! And lastly, switching between different sections drove me crazy when writing on Overleaf. 

Take some time to skim the [Latex-Workshop Wiki](https://github.com/James-Yu/LaTeX-Workshop/wiki/Install) (SyncTex is quite useful). Then, have a look at [Diomidis' Latex Advice](https://github.com/dspinellis/latex-advice/tree/master), which also contains a helpful bibtex schema in case you need to manually construct a citation. 


#### Setting Up 
Brief explanation of how to set up a git project and sync it with both GitHub and Overleaf. 

1. Copy the template folder and name it appropriately. 
2. Enter your paper folder, and run `git init` to create a local repo. 
3. Create a repo on your GitHub profile (optional), and push your local repo to it ([guide](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github#adding-a-local-repository-to-github-using-git)). 

4. add submodule? 

5. Syncing with Overleaf. I presume that 
6. Syncing references from Zotero. There exist [automatic-syncing](https://retorque.re/zotero-better-bibtex/index.html) tools; but seeing as you only do this a few times, you can also manually export your `references.bib` from Zotero to your `tex` folder. 
