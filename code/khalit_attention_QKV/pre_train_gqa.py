import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # or "0,1" for multiple GPUs

import wandb; wandb.login()
from transformers import (
    RobertaForMaskedLM, RobertaConfig, RobertaTokenizerFast,
    GPTNeoForCausalLM, GPTNeoConfig, GPT2TokenizerFast
)
from CustomGQA_GPTNeoConfig import CustomGPTNeoConfig


# #### Models 
# We consider GPT-Neo and BERT as base transformer architectures. This consists of the following blocks linked via residual connections:
# 
# - Embeddings: Map one-hot (sparse) token vectors to a dense vector of length `hidden_size`. Add positional encoding. 
# - $n$ Blocks: Contain self-attention and FFNs.
# - Head: Map hidden state back to a useful output. 
# 
# This specifies some of the model-related hyperparameters. I chose them based on what achieved reasonable performance in the [TinyStories paper](https://arxiv.org/abs/2305.07759), while also being feasible to train on our limited compute budgets. 

# In[5]:


embed_size = 384
kqv_factor = 1
kqv_size = embed_size // kqv_factor
ffn_original_width = 1024
ffn_new_width = int (ffn_original_width + 2 * embed_size * (1 - 1/kqv_factor))

query_groups_factor = 0.75         # try 0.75, 0.5, 0.25, for 8 and 16 heads also 0.125
attn_heads = 16
num_key_value_heads = int(attn_heads * query_groups_factor)
ffn_new_width = ffn_new_width + (kqv_size - int(kqv_size * query_groups_factor))
# ffn_new_width = 1216

config_gpt = dict(

    # EMBEDDING PARAMETERS
    vocab_size              = 10_000,   # number of tokens in the vocabulary 
    hidden_size             = embed_size,      # embedding size (vector length) of each token
    max_position_embeddings = 512,      # maximum sequence length (context window)

    # BLOCKS (ATTN & FFN)
    num_layers          = 3,                    # number of transformer blocks
    attention_types     = [[["global", "local", "global"], 1]], # (GPT-Neo-specific) global and local attention 
    num_heads           = attn_heads,                    # attention heads
    window_size         = 256,                  # (GPT-Neo-specific) for local attention 
    intermediate_size   = 1024,                 # size of 'up-projection' layer in FFN

    pad_token_id = 0,           # need to specify this for tokenizer interop between models
)


config_gqa = dict(

    # EMBEDDING PARAMETERS
    vocab_size              = 10_000,   # number of tokens in the vocabulary
    hidden_size             = embed_size,      # embedding size (vector length) of each token
    max_position_embeddings = 512,      # maximum sequence length (context window)

    # BLOCKS (ATTN & FFN)
    num_layers          = 3,                    # number of transformer blocks
    attention_types     = [[["global", "local", "global"], 1]], # (GPT-Neo-specific) global and local attention
    num_heads           = attn_heads,                    # attention heads
    window_size         = 256,                  # (GPT-Neo-specific) for local attention
    intermediate_size   = ffn_new_width,                 # size of 'up-projection' layer in FFN

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
config_gqa = CustomGPTNeoConfig(num_kv_heads=num_key_value_heads, kqv_size=kqv_size,**config_gqa)


# In[6]:


from GQA_GPTNeoForCausalLM import CustomGPTNeoForCausalLM, GPTNeoGQASelfAttention

# TODO: you should set all pytorch & huggingface seeds here as model initialisation depends on it

gpt = GPTNeoForCausalLM(config=config_gpt)
rob = RobertaForMaskedLM(config=config_rob)

gqa_model_gpt = CustomGPTNeoForCausalLM(config=config_gqa)
# print(gqa_model_gpt.state_dict())


print(f'''
    The custom model has {gqa_model_gpt.num_parameters():,} parameters,
    GPT Neo - {gpt.num_parameters():,} parameters,
    and RoBERTa - {rob.num_parameters():,} parameters.
    ''')

# gpt, rob # uncomment to see model architecture
# gqa_model_gpt


# #### Pre-Processing
# All experiments use the same tokenizer, so in theory, we only need to preprocess our data once. This means you can subsequently skip this CPU-intensive task on the DelftBlue GPU nodes. 
# (except if you are experimenting with different context lengths, in that case you will need to re-tokenise to recover potentially truncated inputs; see below).
# 
# When running experiments on delftblue, it may be fastest to load (parts of) the dataset into memory, as the cluster has pretty slow IO. See `datasets`, specifically [`load_dataset`'s `keep_in_memory` attribute](https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset). 

# In[7]:


import tqdm 
from datasets import load_dataset, DatasetDict

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
tok_rob = get_tokenizer_for_config(RobertaTokenizerFast, config_rob)

from datasets import load_from_disk 
tokenized_dataset = load_from_disk(f'./tokenized_dataset')

train_dataset = tokenized_dataset['train']
eval_dataset  = tokenized_dataset['validation']

assert len(tokenized_dataset['train'][0]['input_ids']) == config_gpt.max_position_embeddings
tokenized_dataset['train'][0]['input_ids'][-10:]
# should be pad tokens (0), given that most short stories are <512 tokens


# #### Training
# Before we get started, you may want to specify which GPU to use. See the first cell in this notebook; make sure to run it before anything else. 

# Huggingface provides some powerful (and often confusingly long) APIs for model training. The `TrainingArguments` specifies our hyperparameters, which are used by the `Trainer` taking in the remaining objects (like `model`, `tokenizer`, and `train_dataset`). Specifically:
# 
# - `learning_rate` and `num_train_epochs` determine how much the model learns. A higher rate is faster, but more unstable. More epochs (entire passes over the dataset) yields incrementally better results, at the cost of more training time. 
# - Batch sizes determine how many samples the model sees in *parallel*. Given `gradient_accumulation_steps=1` and a `batch_size=8`, the model will backpropagate the average loss of 8 samples; if `batch_size=1`, it will average the loss of `gradient_accumulation_steps` samples. It is important to make sure the backpropagated loss is averaged over the same number of samples, when comparing models. 
# 
# - `data_collator` batches (and optionally, pads) the input for the model. We have already padded in our `tokenized_dataset`, and leaving this argument empty will automatically batch the inputs. So why do we need it? 
# 
#     Glad you asked. This has to do with how the loss is computed in causal language modelling. In our case, we try to predict $p(y | x)$, where $x$ is an input sequence of tokens, and $y$ is the next token following that sequence. Our model, unaware of the target token $y$, outputs $\hat y$. 
#     
#     For `Trainer` to compute the (cross-entropy) loss, we need to provide it with both $y$ and $\hat y$. The `DataCollatorForLanguageModeling` knows this, and provides the next token $y$ as a separate part of the input, to the `Trainer`.
# 
#     The loss is the backbone of backpropagation, which we need to actually improve our model. If this is confusing, please re-watch Karpathy's GPT tutorial. 
# 
# If you prefer to write the training loop yourself, check out HF' `run_clm_no_trainer.py` scripts. (`run_mlm_no_trainer.py` for RoBERTa-style masked-language modelling, as opposed to causal language modelling). This can be useful to give you better control over which devices are used for training. 

# In[11]:


from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

def get_hyperparameters(model, dataset, num_key_value_heads=None, kqv_factor=None):
    ''' common hyperparameters to give to the trainer '''

    # TRAINING HYPERPARAMETERS 
    batch_size = 16                  # TinyStories uses 80, but I am training locally on my poor M1 Air
    num_train_epochs = 1             # TinyStories doesn't mention
    gradient_accumulation_steps = 16 # TinyStories uses 16

    lr = 0.001                       # TinyStories uses 5e-4, higher values better for small models

    first_part_id = 'CUSTOM' if isinstance(model, CustomGPTNeoForCausalLM) else 'GPT' if isinstance(model, GPTNeoForCausalLM) else'BERT'

    if first_part_id == 'CUSTOM':
        if num_key_value_heads is not None:
            first_part_id += '-GQA-' + str(query_groups_factor) + 'KV'
        if kqv_factor is not None:
            first_part_id += '-KQV-' + str(kqv_factor) + 'F'

    # future you will thank you for descriptive model names
    # TODO: customise this name such that every model you train has a unique identifier!
    config      = model.config 
    model_name  = '-'.join([
        first_part_id,
        f'{model.num_parameters()//1e6:.1f}M',
        f'{config.num_layers if isinstance(model, GPTNeoForCausalLM) else config.num_hidden_layers}L', 
        f'{config.num_heads if isinstance(model, GPTNeoForCausalLM) else config.num_attention_heads}H', 
        f'{config.hidden_size}C',
        f'{config.intermediate_size}I',
        f'{lr}lr'
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
params_gqa = get_hyperparameters(gqa_model_gpt, train_dataset, num_key_value_heads=num_key_value_heads, kqv_factor=kqv_factor)


# In[12]:


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


# In[13]:


out_dir = './results/models_baseline/' 

trainer_gpt = get_trainer(gpt, tok_gpt, train_dataset, eval_dataset, out_dir, **params_gpt)
trainer_rob = get_trainer(rob, tok_rob, train_dataset, eval_dataset, out_dir, **params_rob)

trainer_gqa_gpt = get_trainer(gqa_model_gpt, tok_gpt, train_dataset, eval_dataset, out_dir, **params_gqa)


# Finally, we can train. 
# 
# This configuration takes ≤24hr to pre-train on my M1 Macbook Air with 16GB RAM. Python takes ≤4GB VRAM at a `batch_size=16` and ≤11GB at `batch_size=64`, though they take the same amount of time to train - likely because this processor is not designed to move that much data in and out of RAM constantly. And tbh, the GPU be lacking. If you decide to go the local-training-route, consider [chai](https://github.com/lvillani/chai) to keep your (Apple) laptop awake – there's probably a windows/linux equivalent too. 

# In[14]:


def do_train(trainer: Trainer, name: str, out_dir: str):
    wandb.init(project='kqv-gqa-gpt-neo', name=name, group='baseline', config=trainer.args)
    trainer.train()
    trainer.save_model(os.path.join(out_dir, name))


# In[15]:


# words vs. tokens 
len(train_dataset['text'][11]), len(train_dataset[11]['input_ids'])
print(params_gqa['model_name'])
trainer_gqa_gpt.model

do_train(trainer_gqa_gpt, params_gqa['model_name'], out_dir)
import torch
model_name = 'CUSTOM-GQA-0.75KV-KQV-1F-8.0M-3L-16H-384C-1120I-0.001lr'
trained_model = trainer_gqa_gpt.model
torch.save(trained_model.state_dict(), f'results/models_baseline/{model_name}/model_state.pt')


# # In[16]:
wandb.finish()



# In[20]:




# Sounds like me after a few beers too many, but at least the grammar is (mostly) correct. The model also learns some basic reasoning-like associations like being 'so high' allows you to see 'the whole world'. 
