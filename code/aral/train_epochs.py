import os, sys, subprocess, torch, wandb, numpy as np, random

# `python train_baselines.py debug` stops wandb sync and behaves like test
# `python train_baselines.py test` uses smol dataset

TEST = len(sys.argv) > 1 and sys.argv[1] == 'test'
DEBUG = len(sys.argv) > 1 and sys.argv[1] == 'debug'
if DEBUG: TEST = True 
else: wandb.login()

# cheeky library i wrote; feel free to use anything else as well. 
from utils import Hyperparams
from common.grid_search import GridSearch, search 
from dataclasses import dataclass
from common.eval_baselines import eval_and_aggregate 

from transformers import (
    GPT2TokenizerFast, GPTNeoConfig, GPTNeoForCausalLM,
    RobertaTokenizerFast, RobertaForMaskedLM, RobertaConfig,
)

@dataclass 
class GPTConfig(GridSearch):

    # EMBEDDING PARAMETERS
    vocab_size              :int = 10_000                   # number of tokens in the vocabulary 
    hidden_size             :int = 256                      # embedding size (vector length) of each token 
    max_position_embeddings :int = 512                      # maximum sequence length (context window)

    # BLOCKS (ATTN & FFN)
    num_layers              :int = 3                        # number of transformer blocks
    attention_types         :int = None                     # (GPT-Neo-specific) global and local attention 
    num_heads               :int = 4                        # attention heads
    window_size             :int = 256                      # (GPT-Neo-specific) for local attention 
    intermediate_size       :int = 1024                     # size of 'up-projection' layer in FFN

    # need to specify this for tokenizer interop between models
    pad_token_id            :int = 0

    def __post_init__(self): 
        if type(self.num_layers) is int:
            layers = ['global', 'local']* (self.num_layers//2 + 1)
            self.attention_types = [[layers[:self.num_layers], 1]]

    def create_model(self):
        return GPTNeoForCausalLM(GPTNeoConfig(**self.__dict__))

    def create_tokenizer(self):
        return GPT2TokenizerFast.from_pretrained('common/10k-tok')

    @property 
    def model_name(self) -> list[str]: 
        ''' things to append to the model name '''
        
        return ['DEBUG'] if DEBUG else [] 

@dataclass
class RoBERTaConfig(GridSearch):

    # EMBEDDING PARAMETERS
    vocab_size              :int = 10_000
    hidden_size             :int = 256
    # we add 1 as RoBERTa uses a special position eearbedding for the padding token (zero vector)
    max_position_embeddings :int = 512 + 1

    # BLOCKS (of course naming is different in roberta :) )
    num_hidden_layers       :int = 3
    num_attention_heads     :int = 4
    intermediate_size       :int = 1024

    pad_token_id            :int = 0

    def create_model(self):
        config = RobertaConfig(**self.__dict__)
        import pdb; pdb.set_trace()
        return RobertaForMaskedLM(config)

    def create_tokenizer(self):
        return RobertaTokenizerFast.from_pretrained('common/10k-tok')

    @property 
    def model_name(self) -> list[str]: 
        ''' things to append to the model name '''
        return ['DEBUG'] if DEBUG else [] 


if __name__ == '__main__': 

    params = Hyperparams(
        model_config=search(GPTConfig(), RoBERTaConfig()),
        debug=DEBUG, 
        group='baseline_epochs',
        num_train_epochs=3,
        batch_size=64,
        gradient_accumulation_steps=4,
    )
    print(params)

    for param in params: 

        print(param)

        score = param.train_and_eval(all_checkpoints=True)

