import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # or "0,1" for multiple GPUs
import torch
import wandb; wandb.login(key = 'b952ac75ca2a31b602f6fa6d434efbc9fbde7a65')
from transformers import (
    RobertaForMaskedLM, RobertaConfig, RobertaTokenizerFast,
    GPTNeoForCausalLM, GPTNeoConfig, GPT2TokenizerFast
)
debug_mode = True

# %% [markdown]
# #### Models 
# We consider GPT-Neo and BERT as base transformer architectures. This consists of the following blocks linked via residual connections:
# 
# - Embeddings: Map one-hot (sparse) token vectors to a dense vector of length `hidden_size`. Add positional encoding. 
# - $n$ Blocks: Contain self-attention and FFNs.
# - Head: Map hidden state back to a useful output. 
# 
# This specifies some of the model-related hyperparameters. I chose them based on what achieved reasonable performance in the [TinyStories paper](https://arxiv.org/abs/2305.07759), while also being feasible to train on our limited compute budgets. 

# %%
config_gpt = dict(

    # EMBEDDING PARAMETERS
    vocab_size              = 10_000,   # number of tokens in the vocabulary 
    hidden_size             = 516,      # embedding size (vector length) of each token 
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
    hidden_size             = 516,      
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

# %%
# TODO: you should set all pytorch & huggingface seeds here as model initialisation depends on it

gpt = GPTNeoForCausalLM(config=config_gpt)
rob = RobertaForMaskedLM(config=config_rob)

print(f'''
    This GPT has {gpt.num_parameters():,} parameters,
     and ROB has {rob.num_parameters():,} parameters.
    ''')

# gpt, rob # uncomment to see model architecture

# %%
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

# %%
from datasets import load_from_disk 

if debug_mode:
    tokenized_dataset = load_from_disk(f'./tokenized_dataset_small')
else:
    tokenized_dataset = load_from_disk(f'./tokenized_dataset')


train_dataset = tokenized_dataset['train']
eval_dataset  = tokenized_dataset['validation']

assert len(tokenized_dataset['train'][0]['input_ids']) == config_gpt.max_position_embeddings
tokenized_dataset['train'][0]['input_ids'][-10:]
# should be pad tokens (0), given that most short stories are <512 tokens

# %% [markdown]
# #### Training
# Before we get started, you may want to specify which GPU to use. See the first cell in this notebook; make sure to run it before anything else. 

# %% [markdown]
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
        'GPT' if isinstance(model, GPTNeoForCausalLM) else 'BERT',
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
out_dir = './results/models_baseline/' 

trainer_gpt = get_trainer(gpt, tok_gpt, train_dataset, eval_dataset, out_dir, **params_gpt)
trainer_rob = get_trainer(rob, tok_rob, train_dataset, eval_dataset, out_dir, **params_rob)

# %% [markdown]
# Finally, we can train. 
# 
# This configuration takes ≤24hr to pre-train on my M1 Macbook Air with 16GB RAM. Python takes ≤4GB VRAM at a `batch_size=16` and ≤11GB at `batch_size=64`, though they take the same amount of time to train - likely because this processor is not designed to move that much data in and out of RAM constantly. And tbh, the GPU be lacking. If you decide to go the local-training-route, consider [chai](https://github.com/lvillani/chai) to keep your (Apple) laptop awake – there's probably a windows/linux equivalent too. 

# %%
def do_train(trainer: Trainer, name: str, out_dir: str): 

    wandb.init(project='tiny-transformers', name=name, group='lauri_baseline', config=trainer.args)
    trainer.train()
    trainer.save_model(os.path.join(out_dir, name))

# %%
# words vs. tokens 
len(train_dataset['text'][11]), len(train_dataset[11]['input_ids'])

# %%
# do_train(trainer_rob, params_rob['model_name'], out_dir)

# %%
do_train(trainer_gpt, params_gpt['model_name'], out_dir) 

wandb.finish()

# %%
do_train(trainer_gpt, params_gpt['model_name'], out_dir) 

trained_model = trainer_gpt.model
model_name = params_gpt['model_name']

torch.save(trained_model.state_dict(), f'./results/models_baseline/{model_name}/model_state.pt')
