# %% [markdown]
# ## Pre-Training [`GPT-Neo`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt_neo/modeling_gpt_neo.py) & [`RoBERTa`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/roberta/modeling_roberta.py) on [`TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories)
# 
# This script shows how to pre-train both models. I put them in one notebook because the majority of the code is shared; but you may want to separate the logic per model. 


# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # or "0,1" for multiple GPUs
from typing import Optional, Tuple
import wandb; wandb.login()
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import (
    RobertaForMaskedLM, RobertaConfig, RobertaTokenizerFast,
    GPTNeoForCausalLM, GPTNeoConfig, GPT2TokenizerFast
)
print(torch.version.cuda)
print(torch.__version__)
print('is avail', torch.cuda.is_available())
print('cur cuda', torch.cuda.current_device())
print("cur device", torch.cuda.get_device_name(0))