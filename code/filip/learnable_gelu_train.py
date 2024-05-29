import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # or "0,1" for multiple GPUs

import wandb; wandb.login()
from transformers import (
    RobertaForMaskedLM, RobertaConfig, RobertaTokenizerFast,
    GPTNeoForCausalLM, GPTNeoConfig, GPT2TokenizerFast, set_seed
)

small_dataset = True
if len(sys.argv) > 1 and (sys.argv[1] == 'True' or sys.argv[1] == 'true'):
    small_dataset = True
    print("Using small dataset")
elif len(sys.argv) > 1 and (sys.argv[1] == 'False' or sys.argv[1] == 'false'):
    small_dataset = False
    print("Using large dataset")
else:
    print("Using small dataset")

config_gpt = dict(

    # EMBEDDING PARAMETERS
    vocab_size              = 10_000,   # number of tokens in the vocabulary 
    hidden_size             = 512,      # embedding size (vector length) of each token 
    max_position_embeddings = 512,      # maximum sequence length (context window)

    # BLOCKS (ATTN & FFN)
    num_layers          = 2,                    # number of transformer blocks
    attention_types     = [[["global", "local"], 1]], # (GPT-Neo-specific) global and local attention 
    num_heads           = 4,                    # attention heads
    window_size         = 256,                  # (GPT-Neo-specific) for local attention 
    intermediate_size   = 1024,                 # size of 'up-projection' layer in FFN

    pad_token_id = 0,           # need to specify this for tokenizer interop between models
)

config_rob = dict(
    
    # EMBEDDING PARAMETERS
    vocab_size              = 10_000,   
    hidden_size             = 512,      
    # we add 1 as RoBERTa uses a special position embedding for the padding token (zero vector)
    max_position_embeddings = config_gpt['max_position_embeddings'] + 1,

    # BLOCKS (of course naming is different in roberta :) )
    num_hidden_layers = config_gpt['num_layers'],
    num_attention_heads = config_gpt['num_heads'],
    intermediate_size=1024,                     

    pad_token_id = 0,
)

config_gpt = GPTNeoConfig(**config_gpt)
config_rob = RobertaConfig(**config_rob)

# implement learnable GELU activation function

import torch
from torch import nn, Tensor
import math


class LearnableGELU(nn.Module):
    def __init__(self, num_parameters: int = 1, init: float = 1.0,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_parameters = num_parameters
        super().__init__()
        self.init = init
        self.beta = nn.Parameter(torch.empty(num_parameters, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.beta, self.init)

    def forward(self, input: Tensor) -> Tensor:
        # Apply the learnable GELU function
        return 0.5 * self.beta * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


# Fix MLP for GPT-Neo with Learnable GELU
class NeoLearnableGELUMLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = nn.Linear(embed_dim, intermediate_size) # to match size for GeGLU
        self.c_proj = nn.Linear(intermediate_size, embed_dim)
        self.act = LearnableGELU(num_parameters=intermediate_size, init=1.0)
        self.dropout = nn.Dropout(float(config.resid_dropout))

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states
    
# Fix MLP for RoBERTa with Learnable GELU
class RobertaLearnableGELUMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = LearnableGELU(num_parameters=config.intermediate_size, init=1.0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class CustomGPTNeoForCausalLM(GPTNeoForCausalLM):
    def __init__(self, config, act_implementation=None):
        super().__init__(config)
        
        if act_implementation == "LEARNABLE_GELU":
            # Override MLP with KAN in each transformer block
            for block in self.transformer.h:
                block.mlp = NeoLearnableGELUMLP(config.intermediate_size, config)

class CustomRobertaForMaskedLM(RobertaForMaskedLM):
    def __init__(self, config, act_implementation=None):
        super().__init__(config)
        
        if act_implementation == "LEARNABLE_GELU":
            # Override MLP with KAN in each transformer block
            for layer in self.roberta.encoder.layer:
                layer.intermediate = RobertaLearnableGELUMLP(config)

import random, numpy as np                
def set_all_seeds(seed=42):

    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_all_seeds()

gpt = CustomGPTNeoForCausalLM(config=config_gpt, act_implementation="LEARNABLE_GELU")
rob = CustomRobertaForMaskedLM(config=config_rob, act_implementation="LEARNABLE_GELU")

print(f'''
    This GPT has {gpt.num_parameters():,} parameters,
     and ROB has {rob.num_parameters():,} parameters.
    ''')

# gpt, rob # uncomment to see model architecture

# %%
from transformers import PreTrainedTokenizer, PretrainedConfig

def get_tokenizer_for_config(Tok: PreTrainedTokenizer, config: PretrainedConfig):

    tokenizer = Tok.from_pretrained(
        '10k-tok',                 # our custom tokenizer
        model_max_length=512       # sequence length (context window)
    )

    # we're using our special tokenizer with only 10'000 tokens instead of 50'256
    assert tokenizer.vocab_size == config.vocab_size

    print(f'padding token is {tokenizer.pad_token}')
    print(f'padding token in config: {config.pad_token_id}, in tokeniser: {tokenizer.pad_token_id}')
    
    return tokenizer 

tok_gpt = get_tokenizer_for_config(GPT2TokenizerFast, config_gpt)
tok_rob = get_tokenizer_for_config(RobertaTokenizerFast, config_rob)

# %%
from datasets import load_from_disk 
tokenized_dataset = load_from_disk(f'./tokenized_dataset_small') if small_dataset else load_from_disk(f'./tokenized_dataset')

train_dataset = tokenized_dataset['train']
eval_dataset  = tokenized_dataset['validation']

assert len(tokenized_dataset['train'][0]['input_ids']) == config_gpt.max_position_embeddings
tokenized_dataset['train'][0]['input_ids'][-10:]
# should be pad tokens (0), given that most short stories are <512 tokens

# %%
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

def get_hyperparameters(model, dataset):
    ''' common hyperparameters to give to the trainer '''

    # TRAINING HYPERPARAMETERS 
    batch_size = 16                  # TinyStories uses 80, but I am training locally on my poor M1 Air
    num_train_epochs = 1             # TinyStories doesn't mention
    gradient_accumulation_steps = 16 # TinyStories uses 16

    lr = 5e-4                        # TinyStories uses 5e-4, higher values better for small models

    # future you will thank you for descriptive model names
    # TODO: customise this name such that every model you train has a unique identifier!
    config      = model.config 
    model_name  = '-'.join([
        'GPT-Learnable-GELU' if isinstance(model, GPTNeoForCausalLM) else 'BERT-Learnable-GELU',
        f'{model.num_parameters()//1e6:.1f}M',
        f'{config.num_layers if isinstance(model, GPTNeoForCausalLM) else config.num_hidden_layers}L', 
        f'{config.num_heads if isinstance(model, GPTNeoForCausalLM) else config.num_attention_heads}H', 
        f'{config.hidden_size}C',
        f'{config.intermediate_size}I'
    ])

    _train_steps = len(dataset) // (batch_size * gradient_accumulation_steps)
    eval_steps = _train_steps // 10 # evaluate every 10% of training steps

    return dict(
        model_name = model_name,
        batch_size = batch_size, 
        num_train_epochs = num_train_epochs,
        gradient_accumulation_steps = gradient_accumulation_steps,
        lr = lr,
        eval_steps = eval_steps
    )

params_gpt = get_hyperparameters(gpt, train_dataset)
params_rob = get_hyperparameters(rob, train_dataset)

# %%
def get_trainer(
        model, tokenizer, train_dataset, eval_dataset, output_dir,
        model_name, batch_size, num_train_epochs, gradient_accumulation_steps, lr, eval_steps):
    ''' more general training arguments you likely want to keep fixed'''

    training_args = TrainingArguments(

        seed       = 42,
        use_cpu    = False, # use GPU if available (not necessarily faster on laptops, but Apple's MPS have good support)

        output_dir = os.path.join(output_dir, model_name),

        # NOTE: training params
        learning_rate    = lr,
        num_train_epochs = num_train_epochs,
        # Use a smaller batch size to fit into GPU RAM. 
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size  = batch_size,
        # You should aim to have the same amount of samples per acc step, in all of your experiments!
        # so, if you increase batch_size, decrease gradient_accumulation_steps by the same factor.
        gradient_accumulation_steps = gradient_accumulation_steps,

        # NOTE: Evaluation params
        # wandb is great for tracking experiments, it will even (try to) save your code nowadays
        evaluation_strategy = 'steps',
        eval_steps = eval_steps,
        save_steps = eval_steps,

        logging_first_step=True,
        logging_steps=100,
        report_to  = 'wandb',
    )

    trainer = Trainer(
        model = model, 
        args = training_args, 
        train_dataset = train_dataset, 
        eval_dataset = eval_dataset,
        data_collator = DataCollatorForLanguageModeling(
            tokenizer, mlm=isinstance(model, RobertaForMaskedLM)),
    )

    # print amount of training steps, and how often the model is evaluated
    print(f'''
    Retrieving Trainer for \033[1m{model_name}\033[0m
        training for {num_train_epochs} epochs, {len(train_dataset)} samples
        {batch_size} batch size, {gradient_accumulation_steps} accumulation steps
        gives {len(train_dataset)//(batch_size * gradient_accumulation_steps)} training steps.
        Evaluating every {eval_steps} steps, {len(eval_dataset)} samples 
        ''')

    return trainer

# %%
out_dir = './results2/models_learnable_gelu/' 

trainer_gpt = get_trainer(gpt, tok_gpt, train_dataset, eval_dataset, out_dir, **params_gpt)
trainer_rob = get_trainer(rob, tok_rob, train_dataset, eval_dataset, out_dir, **params_rob)

# %%
def do_train(trainer: Trainer, name: str, out_dir: str): 

    wandb.init(project='tiny-transformers', name=name, group='learnable_gelu', config=trainer.args)
    trainer.train()
    trainer.save_model(os.path.join(out_dir, name))

    del trainer.model
    del trainer
    wandb.finish()

# %%
# words vs. tokens 
len(train_dataset['text'][11]), len(train_dataset[11]['input_ids'])


# %%
do_train(trainer_gpt, params_gpt['model_name'], out_dir)

do_train(trainer_rob, params_rob['model_name'], out_dir)


