print('Start')
import os, random, torch, numpy as np, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # or "0,1" for multiple GPUs
print(torch.cuda.is_available())
import wandb; wandb.login()
from transformers import (
    RobertaForMaskedLM, RobertaConfig, RobertaTokenizerFast,
    GPTNeoForCausalLM, GPTNeoConfig, GPT2TokenizerFast, set_seed
)
print('Imports done')
small_dataset = True
if len(sys.argv) > 1 and (sys.argv[1] == 'True' or sys.argv[1] == 'true'):
    small_dataset = True
    print("Using small dataset")
elif len(sys.argv) > 1 and (sys.argv[1] == 'False' or sys.argv[1] == 'false'):
    small_dataset = False
    print("Using large dataset")
else:
    print("Using small dataset")

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

# %%
# TODO: you should set all pytorch & huggingface seeds here as model initialisation depends on it
def set_all_seeds(seed=42):

    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_all_seeds()
gpt = GPTNeoForCausalLM(config=config_gpt)
rob = RobertaForMaskedLM(config=config_rob)

print(f'''
    This GPT has {gpt.num_parameters():,} parameters,
     and ROB has {rob.num_parameters():,} parameters.
    ''')

# gpt, rob # uncomment to see model architecture

# %%
# import tqdm 
# from datasets import load_dataset
# 
# dataset = load_dataset('roneneldan/tinystories', num_proc=16)
# 
# n_words = 0 
# for story in tqdm.tqdm(dataset['train']['text']):    # tqdm is used for progress bars around iterables
#     n_words += len(story.split())
# 
# print(f'''
#     Train dataset has {len(dataset['train']['text']):,} samples, 
#     totalling {n_words:,} words (avg. {n_words/len(dataset['train']['text']):,.1f} per story).
#     ''')
# 
# %% [markdown]
# #### Tokenisation
# Two things happen during tokenisation, of which the latter is optional:
# 
# 1. Map (sub-)words to integers $ \in [0, 10000]$.
# 2. Truncate/pad each input as necessary to fit within `hidden_size`. 
# 
# All it does is map frequent character strings (i.e. tokens) to integers, which it can then use to look up the corresponding embedding vector for that integer index.
# 
# The intuition is that by assigning each token its own embedding (vector of `1 x hidden_size`) in the transformer, will help it learn semantical understanding of these tokens. E.g. say that you have two tokens 'about' and 'around', the transformer should learn similar embedding vectors for each. 

# %%
from transformers import PreTrainedTokenizer, PretrainedConfig
print('get tokenizers')
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

'''
# %%
# NOTE: uncomment to tokenize the dataset (both tokenizers should give identical results)

tokenized_gpt = dataset.map(
    lambda x: tok_gpt(x['text'], truncation=True, padding='max_length'), 
    batched=True, num_proc=64, batch_size=1_000)

tokenized_gpt.save_to_disk('./tokenized_dataset', num_proc=5)

# # sanity check that the tokenisers are indeed identical 
# tokenized_rob = dataset.map(
#     lambda x: tok_rob(x['text'], truncation=True, padding='max_length'), 
#     batched=True, num_proc=64, batch_size=1_000)

# for split in ['train', 'validation']:
#     for sample_gpt, sample_rob in tqdm.tqdm(zip(tokenized_gpt[split], tokenized_rob[split])):
#         assert sample_gpt == sample_rob

# %%
'''
from datasets import load_from_disk 
tokenized_dataset = load_from_disk(f'./tokenized_dataset_small') if small_dataset else load_from_disk(f'./tokenized_dataset')

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
        'GPT-base' if isinstance(model, GPTNeoForCausalLM) else 'BERT-base',
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

    wandb.init(project='tiny-transformers', name=name, group='baseline', config=trainer.args)
    trainer.train()
    trainer.save_model(os.path.join(out_dir, name))

    del trainer.model
    del trainer
    wandb.finish()

# %%
# words vs. tokens 
len(train_dataset['text'][11]), len(train_dataset[11]['input_ids'])

# %%
do_train(trainer_rob, params_rob['model_name'], out_dir)

# %%
do_train(trainer_gpt, params_gpt['model_name'], out_dir)