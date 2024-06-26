<h2 align="center"><b><h3>Evaluating Adaptive Activation Functions in Language Modelsn</h3></b></h2>


<p align="center">
  <b>Filip Ignijic</b>
  <b> Supervisor:  Aral de Moor, Responsible Professors: Maliheh Izadi, Arie van Deursen </b>
</p>

<p align="center">
  <i>
    Delft University of Technology<br>
    CSE3000 Research Project EEMCS, Delft University of Technology, The Netherlands<br>
  </i>
</p>
<br>

<p align="center">
  <a href="https://www.overleaf.com/read/bfkdnmbgkjdv#65d3fa"><b>Paper (update with link)</b></a><br>
  <a href="https://huggingface.co/collections/AISE-TUDelft/brp-tiny-transformers-666c352b3b570f44d7d2a519"><b>HuggingFace Model Collection</b></a>
</p>
<br>

---
<br>
<h3 align="center"><b>Abstract</b></h3><br>
The rapid expansion of large language models (LLMs) driven by the transformer architecture has raised concerns about the lack of high-quality training data. This study investigates the role of activation functions in smaller-scale language models, specifically those with approximately 10M parameters, to ensure sustained progress in LLM development despite data limitations. Activation functions, crucial for neural network performance, have evolved significantly, but comprehensive comparisons under consistent conditions remain scarce, especially for smaller parameter count models. This research systematically evaluates traditional and novel activation functions, including learnable variants, and introduces the Kolmogorov-Arnold Network (KAN) to language modeling. Using Hugging Face implementations of GPT-Neo and RoBERTa models, performance impacts were assessed through the BabyLM evaluation pipeline. 
<br>
The results indicate that activation functions do not significantly impact the performance of these models. Additionally, the model with the KAN network underperformed compared to models with traditional architectures in the context of this study. These findings suggest that optimizing activation functions may not be crucial for smaller language models, emphasizing the need for further research to explore other architectural improvements.
</br>

---
<br>

### Contents

This repository contains the (online) appendix, source code used to train our models, and the offline evaluation. 
This repository contains replication code for the paper "Evaluating Adaptive Activation Functions in Language Models: Does choice of activation function matter in smaller Langaunge Models?".

## Installation
Make sure you have cuda installed. This code was tested with cuda 12.1 and 11.8

```bash
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
Note: install torch for cuda version on your machine

## Dataset
The dataset should be placed in tokenized_dataset folder. 


## Replication

1. First run one of the pre-train files to pre-train the model. Each of the '***_train.py' files will pre-train GPT-Neo and roBERTa models with that activation function.
   1. It takes 2 arguments, the first is boolean for using small dataset or not, the second one is seed you want to use for pre-training.
2. Navigate to `common` folder and run `eval_baselines.py` where you have to set `MODEL_DIR` in line 5 to the path of the folder with models you want to evalute.

### `***_train.py` files
1. Set the configuration files for roBERTa and NEo
2. Load the datasets
3. Initialize the trainers
4. Pretrain and save the models



### `evaluate_baselines.py`
1. Evaluates the models based on babyLM pipeline 2023
2. Aggregates the reults based on babyLM challange
3. See https://github.com/babylm/evaluation-pipeline-2023 for details
