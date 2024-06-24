import os
from datasets import load_from_disk

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # or "0,1" for multiple GPUs

import wandb
import sys
import tqdm 
import pdb
from datasets import load_dataset

wandb.login(key="6f46f55bd51d76400f1e877ea7dfa75c5c7d05d6")
from transformers import (
    RobertaForMaskedLM, RobertaConfig, RobertaTokenizerFast, AlbertTokenizerFast,
    GPTNeoForCausalLM, GPTNeoConfig, GPT2TokenizerFast, set_seed,
    PreTrainedTokenizer, PretrainedConfig,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
import os, torch, wandb, numpy as np, random


def set_all_seeds(seed=42):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_tokenizer_for_config(Tok: PreTrainedTokenizer, config: PretrainedConfig, tokeniser_path):

    tokenizer = Tok.from_pretrained(
        tokeniser_path,                                         # our custom tokenizer
        model_max_length=config.max_position_embeddings    # sequence length (context window)
    )

    # assert tokenizer.vocab_size == config.vocab_size

    print(f'padding token is {tokenizer.pad_token}')
    print(f'padding token in config: {config.pad_token_id}, in tokeniser: {tokenizer.pad_token_id}')
    
    return tokenizer 


def get_hyperparameters(model, dataset, tokeniser_name):
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
        f'{tokeniser_name}',
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


def get_trainer(
        model, tokenizer, train_dataset, eval_dataset, output_dir,
        model_name, batch_size, num_train_epochs, gradient_accumulation_steps, lr, eval_steps):
    ''' more general training arguments you likely want to keep fixed'''

    training_args = TrainingArguments(

        seed=42,
        use_cpu=False,  # use GPU if available (not necessarily faster on laptops, but Apple's MPS have good support)

        output_dir=os.path.join(output_dir, model_name),

        # NOTE: training params
        learning_rate=lr,
        num_train_epochs=num_train_epochs,
        # Use a smaller batch size to fit into GPU RAM.
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        # You should aim to have the same amount of samples per acc step, in all of your experiments!
        # so, if you increase batch_size, decrease gradient_accumulation_steps by the same factor.
        gradient_accumulation_steps=gradient_accumulation_steps,

        # NOTE: Evaluation params
        # wandb is great for tracking experiments, it will even (try to) save your code nowadays
        evaluation_strategy='steps',
        eval_steps=eval_steps,
        save_steps=eval_steps,

        logging_first_step=True,
        logging_steps=100,
        report_to='wandb',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer, mlm=isinstance(model, RobertaForMaskedLM)),
    )

    # print amount of training steps, and how often the model is evaluated
    print(f'''
    Retrieving Trainer for \033[1m{model_name}\033[0m
        training for {num_train_epochs} epochs, {len(train_dataset)} samples
        {batch_size} batch size, {gradient_accumulation_steps} accumulation steps
        gives {len(train_dataset) // (batch_size * gradient_accumulation_steps)} training steps.
        Evaluating every {eval_steps} steps, {len(eval_dataset)} samples 
        ''')

    return trainer


def do_train(trainer: Trainer, name: str, out_dir: str):
    wandb.init(project='tiny-transformers', name=name, group='baseline', config=trainer.args)
    trainer.train()
    trainer.save_model(os.path.join(out_dir, name))

def tokenize_dataset(tok):
    dataset = load_dataset('roneneldan/tinystories', num_proc=16)

    n_words = 0 
    for story in tqdm.tqdm(dataset['train']['text']):    # tqdm is used for progress bars around iterables
        n_words += len(story.split())

    # def tokenize_function(example):
    #     # Use encode to tokenize the    
    #     encoded = tok(
    #         example['text'],  # Assuming the text column is named 'text'
    #         padding='max_length',  # Pad to max length
    #         truncation=True  # Truncate if the text is longer than max length
    #     )
    #     return encoded

    # tokenized_gpt = dataset.map(
    #     tokenize_function, 
    #     batched=True, 
    #     num_proc=32, 
    #     batch_size=1_000
    # )

    def tokenize_function(examples):
        return tok(examples['text'], truncation=True, padding='max_length', add_special_tokens=False)

    tokenized_gpt = dataset.map(tokenize_function, batched=True, num_proc=32, batch_size=1000)

    tokenized_gpt.save_to_disk('./tokenized_dataset', num_proc=5)


if __name__ == "__main__":
    tokeniser_path = sys.argv[1]
    tokeniser_name = sys.argv[2]
    given_hidden_size = int(sys.argv[3])
    given_vocab_size = int(sys.argv[4])
    model = sys.argv[5]

    set_all_seeds()

    config_gpt = dict(

        # EMBEDDING PARAMETERS
        vocab_size              = given_vocab_size,   # number of tokens in the vocabulary 
        hidden_size             = given_hidden_size,      # embedding size (vector length) of each token 
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
        vocab_size              = given_vocab_size,   
        hidden_size             = given_hidden_size,      
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

    gpt = GPTNeoForCausalLM(config=config_gpt)
    rob = RobertaForMaskedLM(config=config_rob)

    if "sp" in tokeniser_name:
        tok_gpt = get_tokenizer_for_config(AlbertTokenizerFast, config_gpt, tokeniser_path)
        tok_rob = get_tokenizer_for_config(AlbertTokenizerFast, config_rob, tokeniser_path)
    else:
        tok_gpt = get_tokenizer_for_config(RobertaTokenizerFast, config_gpt, tokeniser_path)
        tok_rob = get_tokenizer_for_config(RobertaTokenizerFast, config_rob, tokeniser_path)        

    tokenize_dataset(tok_gpt)

    tokenized_dataset = load_from_disk(f'./tokenized_dataset')

    train_dataset = tokenized_dataset['train']
    eval_dataset = tokenized_dataset['validation']

    # print(tokenized_dataset['train'][0])
    assert len(tokenized_dataset['train'][0]['input_ids']) == config_gpt.max_position_embeddings

    params_gpt = get_hyperparameters(gpt, train_dataset, tokeniser_name)
    params_rob = get_hyperparameters(rob, train_dataset, tokeniser_name)

    out_dir = './results/models_sp/'

    trainer_gpt = get_trainer(gpt, tok_gpt, train_dataset, eval_dataset, out_dir, **params_gpt)
    trainer_rob = get_trainer(rob, tok_rob, train_dataset, eval_dataset, out_dir, **params_rob)

    if model == "GPT":
        do_train(trainer_gpt, params_gpt['model_name'], out_dir)
    else:
        do_train(trainer_rob, params_rob['model_name'], out_dir)