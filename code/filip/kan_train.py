import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # or "0,1" for multiple GPUs


import wandb; wandb.login()

# TODO: Some gpu error, this fixes it but ask aral for the details
# import torch
# torch.use_deterministic_algorithms(False)

small_dataset = True
if len(sys.argv) > 1 and sys.argv[1] and sys.argv[1] == "true":
    small_dataset = True
elif len(sys.argv) > 1 and sys.argv[1] and sys.argv[1] == "false":
    small_dataset = False

from transformers import (
    GPTNeoForCausalLM, GPTNeoConfig, GPT2TokenizerFast, set_seed
)

# %%
config_gpt = dict(

    # EMBEDDING PARAMETERS
    vocab_size              = 10_000,   # number of tokens in the vocabulary 
    hidden_size             = 256,      # to get to 10m parameters
    max_position_embeddings = 512,      # maximum sequence length (context window)

    # BLOCKS (ATTN & FFN)
    num_layers=2,                               # number of transformer blocks
    attention_types=[[["global", "local"], 1]], # (GPT-Neo-specific) global and local attention 
    num_heads=4,                                # attention heads
    window_size=256,                            # (GPT-Neo-specific) for local attention 
    intermediate_size=1024,                     # size of 'up-projection' layer in FFN

    pad_token_id = 0,           # need to specify this for tokenizer interop between models
)

config_gpt = GPTNeoConfig(**config_gpt)

# %% [markdown]
# ### Implement KAN instead of MLP

# %%
from efficient_kan.model import KAN
import torch
from torch import nn
class KANtoMLP(nn.Module):

    def __init__(self, hidden_size, inner_dim, config):
        super().__init__()
        self.c_fc = KAN([hidden_size, inner_dim])
        self.c_proj = KAN([inner_dim, hidden_size])
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.resid_dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
        
import random, numpy as np

class CustomGPTNeoForCausalLM(GPTNeoForCausalLM):
    def __init__(self, config, mlp_implementation=None):
        super().__init__(config)
        self.attn_implementation = mlp_implementation
        
        if mlp_implementation == "KAN":
            # Override MLP with KAN in each transformer block
            for block in self.transformer.h:
                block.mlp = KANtoMLP(config.hidden_size, config.intermediate_size, config)
def set_all_seeds(seed=42):

    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_all_seeds()

model = CustomGPTNeoForCausalLM(config=config_gpt, mlp_implementation="KAN")

print(f'''
    This GPT has {model.num_parameters():,} parameters,
    ''')

from transformers import PreTrainedTokenizer, PretrainedConfig

def get_tokenizer_for_config(Tok: PreTrainedTokenizer, config: PretrainedConfig):

    tokenizer = Tok.from_pretrained(
        '10k-tok',                                         # our custom tokenizer
        model_max_length=config.max_position_embeddings    # sequence length (context window)
    )

    # we're using our special tokenizer with only 10'000 tokens instead of 50'256
    assert tokenizer.vocab_size == config.vocab_size

    print(f'padding token is {tokenizer.pad_token}')
    print(f'padding token in config: {config.pad_token_id}, in tokeniser: {tokenizer.pad_token_id}')
    
    return tokenizer 

tok_gpt = get_tokenizer_for_config(GPT2TokenizerFast, config_gpt)


# %%
from datasets import load_from_disk 
tokenized_dataset = load_from_disk(f'./tokenized_dataset_small') if small_dataset else load_from_disk('./tokenized_dataset')

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

    config      = model.config 
    model_name  = '-'.join([
        'KAN-GPT' if isinstance(model, CustomGPTNeoForCausalLM) else 'BERT',
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

params_gpt = get_hyperparameters(model, train_dataset)

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
            tokenizer, mlm=False),
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
out_dir = './results/models_kan' 

trainer_gpt = get_trainer(model, tok_gpt, train_dataset, eval_dataset, out_dir, **params_gpt)

# %% [markdown]
# Finally, we can train. 
# 
# This configuration takes ≤24hr to pre-train on my M1 Macbook Air with 16GB RAM. Python takes ≤4GB VRAM at a `batch_size=16` and ≤11GB at `batch_size=64`, though they take the same amount of time to train - likely because this processor is not designed to move that much data in and out of RAM constantly. And tbh, the GPU be lacking. If you decide to go the local-training-route, consider [chai](https://github.com/lvillani/chai) to keep your (Apple) laptop awake – there's probably a windows/linux equivalent too. 

# %%
def do_train(trainer: Trainer, name: str, out_dir: str): 

    wandb.init(project='tiny-transformers', name=name, group='KAN', config=trainer.args)
    trainer.train()
    trainer.save_model(os.path.join(out_dir, name))

# %%
# words vs. tokens 
len(train_dataset['text'][11]), len(train_dataset[11]['input_ids'])

# %%
do_train(trainer_gpt, params_gpt['model_name'], out_dir)