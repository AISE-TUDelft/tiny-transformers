import os, sys, subprocess, torch, wandb, numpy as np, random

# `python train_baselines.py debug` stops wandb sync and behaves like test
# `python train_baselines.py test` uses smol dataset

TEST = len(sys.argv) > 1 and sys.argv[1] == 'test'
DEBUG = len(sys.argv) > 1 and sys.argv[1] == 'debug'
if DEBUG: TEST = True 

from typing import Any
from datasets import load_from_disk 
from transformers import (
    RobertaForMaskedLM, RobertaConfig, RobertaTokenizerFast,
    GPTNeoForCausalLM, GPTNeoConfig, GPT2TokenizerFast,
    PreTrainedTokenizer, PretrainedConfig,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling,
    set_seed
)


MODEL_DIR = 'models/baseline'

for model in os.listdir(MODEL_DIR):

    # print([f for f in os.listdir('.')])
    model_path = os.path.join(os.path.abspath(MODEL_DIR), model)

    with open(os.path.join(model_path, 'eval.log'), 'wb') as f:
        process = subprocess.Popen(f'CUDA_VISIBLE_DEVICES=0 ./evaluate.sh {model_path}', 
                stdout=subprocess.PIPE, shell=True)
        for c in iter(lambda: process.stdout.read(1), b''):
            sys.stdout.buffer.write(c)
            f.buffer.write(c)

    break

