print("started")
import wandb; wandb.login()
import sys
from transformers import GPT2TokenizerFast, GPTNeoForCausalLM, GPTNeoConfig
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset, load_from_disk, DatasetDict

# debug or local debug
small_dataset = True
if len(sys.argv) > 1 and (sys.argv[1] == 'True' or sys.argv[1] == 'true'):
    small_dataset = True
    print("Using small dataset")

config = GPTNeoConfig(

    # number of tokens in the vocabulary 
    vocab_size = 10_000, 
    # embedding size (vector length) of each token 
    hidden_size=512, 
    # we thus have an embedding block of 512 x 10'000 parameters

    # maximum sequence length, though inputs longer than `hidden_size` will be iteratively processed
    max_position_embeddings = 512, 

    # number of transformer blocks. div by 2 for attention_types
    num_layers=2, 
    # for global and local attention (GPT-Neo-specific)
    attention_types=[[["global", "local"], 1]], 

    num_heads=4,     # attention heads
    window_size=256, # for local attention (GPT-Neo-specific)

    intermediate_size=1024, # size of 'up-projection' layer in FFN
)
# config
tokenizer = GPT2TokenizerFast.from_pretrained('10k-gpt-neo', model_max_length=config.hidden_size)
assert tokenizer.model_max_length == 512 
assert tokenizer.vocab_size == config.vocab_size

# printing this because of a bug in tokenizers (should be fixed now) https://github.com/huggingface/transformers/issues/26500
print(f'padding token is {tokenizer.pad_token}')
# HF wasn't saving this nor the tokenizer's pad_token
config.pad_token_id = tokenizer.pad_token_id

# %%
dataset = load_dataset("roneneldan/TinyStories")

small_dataset = DatasetDict({
    'train': dataset['train'].select(range(1000)),
    'validation': dataset['validation'].select(range(100))
})

dataset = small_dataset

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length')

tokenized_dataset = dataset.map(lambda x: tokenizer(x['text'], truncation=True, padding='max_length'), 
     batched=True, num_proc=None, batch_size=1_000)
tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=None, batch_size=1000)

tokenized_dataset.save_to_disk(f'./tokenized_dataset_small', num_proc=None)
print(small_dataset)
tokenized_dataset = load_from_disk(f'./tokenized_dataset_small') if small_dataset else load_from_disk(f'./tokenized_dataset')
#tokenized_dataset = load_from_disk(f'./tokenized_dataset')
print(tokenized_dataset.shape)

assert len(tokenized_dataset['train'][0]['input_ids']) == config.hidden_size
tokenized_dataset['train'][0]['input_ids'][-10:] 
# should be eos tokens (9999), given that most short stories are <512 tokens

# MODEL TIME
# %%
model = GPTNeoForCausalLM(config=config)

print(f'The model has {model.num_parameters():,} parameters.')
# Token embeddings:    10'000 * 512 = 5M 
# Position embeddings:    512 * 512 = 260K
# 2 Blocks: 2 * (~1M + ~1M) = 4M
#   Attention: 4 * 512 * 512  (4 because of k, q, v, o)     = 1M 
#   FFNs:      2 * 512 * 1024 (c_fc, c_proj)                = 1M 

# total: ~9.5M 

# note that the above does not account for the LM_head, which typically gets swapped out depending on the downstream task. 
# the LM_head, in this case, maps logits back to tokens
# 512 * 10'000 = 5M    (A decent chunk compared to our 9.5M parameters)

# NOTE: uncomment to see the model architecture 
# model 

# %%
# for comparison. Note, wte should be (10000, 64)
# GPTNeoForCausalLM.from_pretrained('roneneldan/TinyStories-1M')
train_dataset, eval_dataset = tokenized_dataset['train'], tokenized_dataset['validation']

batch_size = 16 # TinyStories claims 80, but I am training locally on my poor M1 Air
num_train_epochs = 1  # TinyStories doesn't mention
gradient_accumulation_steps = 16 # TinyStories claims 16

lr = 5e-4 # TinyStories claims 5e-4, higher values are preferable for smaller models

_train_steps = len(train_dataset) // (batch_size * gradient_accumulation_steps)
eval_steps = _train_steps // 10 # evaluate every 10% of training steps

model_name = f'Base-GPT-Neo-small-dataset{model.num_parameters()//1e6:.1f}M-{config.num_layers}L-{config.num_heads}H-{config.hidden_size}C-{config.intermediate_size}I'

# %%
print(eval_steps)
training_args = TrainingArguments(

    seed       = 42,
    use_cpu    = False, # use GPU if available (not necessarily faster on laptops, but Apple's MPS have good support)
    output_dir = f'./results/models/{model_name}',

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
    logging_steps=max(1,eval_steps),
    report_to  = 'wandb',
)

trainer = Trainer(
    model = model, 
    args = training_args, 
    train_dataset = train_dataset, 
    eval_dataset = eval_dataset,
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# print amount of training steps, and how often the model is evaluated
print(f'''
    training for {num_train_epochs} epochs, {len(train_dataset)} samples
    {batch_size} batch size, {gradient_accumulation_steps} accumulation steps
    gives {_train_steps} training steps.

    evaluating every {eval_steps} steps, {len(eval_dataset)} samples 
    ''')

# %% [markdown]
# Finally, we can train. 
# 
# This configuration takes ≤24hr to pre-train on my M1 Macbook Air with 16GB RAM. Python takes ≤4GB VRAM at a `batch_size=16` and ≤11GB at `batch_size=64`, though they take the same amount of time to train - likely because this processor is not designed to move that much data in and out of RAM constantly. And tbh, the GPU be lacking. If you decide to go the local-training-route, consider [chai](https://github.com/lvillani/chai) to keep your (Apple) laptop awake – there's probably a windows/linux equivalent too. 

# %%
wandb.init(project='tinystories', name=model_name, config=training_args)
trainer.train()
trainer.save_model(f'./results/models/{model_name}')

# %% [markdown]
# ## Inference!
# Try out your own pre-trained model on some prompts

# %%
import textwrap # for pretty printing
w = textwrap.TextWrapper(replace_whitespace=False, break_long_words=False, width=60, initial_indent='   ', subsequent_indent='  ')
def see(text): print('\n\033[3m' + '\n\n'.join(['\n'.join(w.wrap(line))
                 for line in text.splitlines() if line.strip() != '']) + '\033[0m\n')

# %%
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = 'Base-GPT-Neo-small-dataset9.0M-2L-4H-512C-1024I'
tokenizer = AutoTokenizer.from_pretrained('10k-gpt-neo')
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(f'results/models/{model_name}')

# %%
prompt = 'One weekend Filip had a '
input_ids = tokenizer.encode(prompt, return_tensors='pt')

output = model.generate(input_ids, max_length=300, num_beams=1)
output_text = tokenizer.decode(output[0])

# textwrap with indentation on every new paragraph
see(output_text)


