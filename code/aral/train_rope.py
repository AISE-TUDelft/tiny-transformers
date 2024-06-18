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

from modeling_gpt_neo import GPTNeoConfig, GPTNeoForCausalLM
from transformers import GPT2TokenizerFast

@dataclass 
class GPTConfig(GridSearch):

    # EMBEDDING PARAMETERS
    vocab_size              :int = 10_000                   # number of tokens in the vocabulary 
    hidden_size             :int = 256                      # embedding size (vector length) of each token 
    max_position_embeddings :int = search(64, 512)          # maximum sequence length (context window)

    # BLOCKS (ATTN & FFN)
    num_layers              :int = 3                        # number of transformer blocks
    attention_types         :int = None                     # (GPT-Neo-specific) global and local attention 
    num_heads               :int = 4                        # attention heads
    window_size             :int = 256                      # (GPT-Neo-specific) for local attention 
    intermediate_size       :int = 1024                     # size of 'up-projection' layer in FFN

    # need to specify this for tokenizer interop between models
    pad_token_id            :int = 0
    rope_scaling     :GridSearch = search(
        None,
        {'type': 'linear', 'factor': 2.0},
        {'type': 'dynamic', 'factor': 2.0},
    )

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
        
        rope_config = [
            self.rope_scaling['type'],
            f'{self.rope_scaling["factor"]}a', 
        ] if self.rope_scaling is not None else []

        return rope_config \
            + [f'{self.max_position_embeddings}PE'] \
            + ['DEBUG'] if DEBUG else [] 

if __name__ == '__main__': 

    params = Hyperparams(
        model_config=GPTConfig(), 
        debug=DEBUG, 
        group='rope'
    )
    print(params)
    
    for param in params: 
        print(param)
        param.train_and_eval()
    
    