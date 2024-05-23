import json
import os
import transformers
import torch
import numpy as np
from datasets import load_from_disk
from transformers import (
    RobertaForMaskedLM, RobertaConfig, RobertaTokenizerFast,
    GPTNeoForCausalLM, GPTNeoConfig, GPT2TokenizerFast,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling,
)
from sparse_gpt_neo import SparseGPTNeoForCausalLM
from configuration_sparse_gpt_neo import SparseGPTNeoConfig
from sparse_roberta import SparseRobertaForMaskedLM
from configuration_sparse_roberta import SparseRobertaConfig
#from common.eval_baselines import eval_and_aggregate
import wandb


import argparse

# set seed
def set_seeds(seed=123):
    np.random.seed(seed)
    transformers.set_seed(seed)
    torch.cuda.manual_seed_all(seed)

# load training data
def load_training_data(path="data/TokenizedTinyStories"):
    return load_from_disk(path)

# load model according to config

def get_tokenizer_for_config(tok, config):
    tokenizer = tok.from_pretrained(
        'code/common/10k-tok',  # our custom tokenizer
        model_max_length=512  # sequence length (context window)
    )

    # we're using our special tokenizer with only 10'000 tokens instead of 50'256
    assert tokenizer.vocab_size == config.vocab_size

    print(f'padding token is {tokenizer.pad_token}')
    print(f'padding token in config: {config.pad_token_id}, in tokeniser: {tokenizer.pad_token_id}')

    return tokenizer
def load_tokenizer_and_model(model_type, sparsity_type, sparsity_level):
    model_name = ""
    if model_type == 'gpt':
        if sparsity_type == 'baseline':
            with open("code/eugene/model_configs/gpt_baseline.json",'r') as config_file:
                config_dict = json.load(config_file)
            config_gpt = GPTNeoConfig(**config_dict)
            tok_gpt = get_tokenizer_for_config(GPT2TokenizerFast,config_gpt)
            return tok_gpt, GPTNeoForCausalLM(config=config_gpt)
        else:
            with open(f"code/eugene/model_configs/gpt_{sparsity_type}_{sparsity_level}.json", 'r') as config_file:
                config_dict = json.load(config_file)
            config_gpt = SparseGPTNeoConfig(**config_dict)
            tok_gpt = get_tokenizer_for_config(GPT2TokenizerFast,config_gpt)
            return tok_gpt, SparseGPTNeoForCausalLM(config=config_gpt)

    elif model_type == 'roberta':
        if sparsity_type == 'baseline':
            with open("code/eugene/model_configs/roberta_baseline.json", 'r') as config_file:
                config_dict = json.load(config_file)
            config_rob = RobertaConfig(**config_dict)
            tok_rob = get_tokenizer_for_config(RobertaTokenizerFast,config_rob)
            return tok_rob, RobertaForMaskedLM(config=config_rob)
        else:
            with open(f"code/eugene/model_configs/roberta_{sparsity_type}_{sparsity_level}.json", 'r') as config_file:
                config_dict = json.load(config_file)
            config_rob = SparseRobertaConfig(**config_dict)
            tok_rob = get_tokenizer_for_config(RobertaTokenizerFast,config_rob)
            model = SparseRobertaForMaskedLM(config=config_rob)
            #print(tok_rob, model)
            return tok_rob, model
    else:
        raise Exception("unknown model")

# training loop
def get_model_name(model_type, sparsity_type, sparsity_level):
    if sparsity_type == 'baseline':
        return f"{model_type}_{sparsity_type}"
    else:
        return f"{model_type}_{sparsity_type}_{sparsity_level}"



def train(name, model, tokenizer, dataset, debug, gpu):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu) # or "0,1" for multiple GPUs
    lr = 5e-4
    num_epochs = 2
    batch_size = 32 if not debug else 8
    grad_accum_steps = 16

    output_dir = f'models/eugene/{name}'



    training_data = dataset['train'] if not debug else dataset['train'].select(range(batch_size*10))
    validation_data = dataset['validation'] if not debug else dataset['validation'].select(range(batch_size*4))
    num_training_steps = len(dataset['train']) // (batch_size * grad_accum_steps)
    num_eval_steps = num_training_steps // 10

    training_args = TrainingArguments(
        seed=123,
        use_cpu=False,
        learning_rate=lr,
        output_dir=f'{output_dir}/checkpoints',
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,

        eval_strategy='steps',
        eval_steps=num_eval_steps if not debug else 5,
        save_steps=num_eval_steps if not debug else 5,

        logging_first_step=True,
        logging_steps=num_eval_steps if not debug else 1,
        report_to='wandb' if not debug else 'none',
    )


    trainer = Trainer(

        model=model.cuda(),
        args=training_args,

        train_dataset=training_data,
        eval_dataset=validation_data,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer, mlm=isinstance(tokenizer, RobertaForMaskedLM)),
    )


    if not debug:
        model_type = 'gpt' if 'gpt' in name else 'roberta'
        with open('code/eugene/wandb_key.txt') as f:
            wandb.login(key=f.read())
        wandb.init(
            entity='tiny-transformers', project='eugene',
            group=model_type, name=name,
           )
    else:
        print('\033[1mRUNNING IN DEBUG MODE \033[0m')

    if os.path.exists(output_dir) and os.path.exists(f"{output_dir}/model.safetensors"):
        print(f'\033[1m{name} has already been trained; skipping pretraining\033[0m')
    else:
        trainer.train()
        trainer.save_model(output_dir)

    # evaluation, whose pipeline i dont have time figuring out at the moment
    # set_seeds()
    # score = eval_and_aggregate(model_path=output_dir, index=gpu)

    # if not debug:
    #     wandb.log(score)
    #     wandb.finish()
    # print(score)

    #



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate specified model")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode quick test (limits max steps).")
    parser.add_argument('--gpu', type=int, required=True, help="Specify the GPU number to use.")
    parser.add_argument('--model_type', type=str, required=True, choices=['gpt', 'roberta'],
                        help="Specify the model type.")
    parser.add_argument('--sparsity_type', type=str, required=True, choices=['baseline', 'moe', 'cnt', 'pkm'],
                        help="Specify the sparsity type.")
    parser.add_argument('--sparsity_level', type=str, required=True, choices=['low', 'medium', 'high'],
                        help="Specify the sparsity level.")

    assert torch.cuda.is_available(), "no cuda"

    args = parser.parse_args()

    set_seeds()
    dataset = load_training_data()


    tokenizer, model = load_tokenizer_and_model(args.model_type, args.sparsity_type, args.sparsity_level)
    mode_name = get_model_name(args.model_type, args.sparsity_type, args.sparsity_level)
    train(mode_name, model, tokenizer, dataset, args.debug, args.gpu)

