import os, sys, subprocess, torch, wandb, numpy as np, random

# `python train_baselines.py debug` stops wandb sync and behaves like test
# `python train_baselines.py test` uses smol dataset

TEST = len(sys.argv) > 1 and sys.argv[1] == 'test'
DEBUG = len(sys.argv) > 1 and sys.argv[1] == 'debug'
if DEBUG: TEST = True 
else: wandb.login()

from typing import Any
from datasets import load_from_disk 
from transformers import (
    RobertaForMaskedLM, RobertaConfig, RobertaTokenizerFast,
    GPTNeoForCausalLM, GPTNeoConfig, GPT2TokenizerFast,
    PreTrainedTokenizer, PretrainedConfig,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling,
    set_seed
)

# cheeky library i wrote; feel free to use anything else as well. 
from grid_search import GridSearch, search 
from dataclasses import dataclass
from eval_baselines import eval_and_aggregate

@dataclass 
class Gpt(GridSearch):
    ''' extend GridSearch with dataclass to search the values in `search`,
        can be used recursively; but you need to annotate your types! 

        (a pretty fucking arbitrary constraint from dataclasses in a non-typed language 
        especially if types can be directly inferred; but what do i know)
    '''

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
        # we need this for the values that depend on searched values in our config 

        # first make sure we're in an instantiated (iterated) object 
        if type(self.num_layers) is int:
            layers = ['global', 'local']* (self.num_layers//2 + 1)
            self.attention_types = [[layers[:self.num_layers], 1]]

@dataclass
class Rob(GridSearch):

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


@dataclass 
class Hyperparams(GridSearch):

    dataset                     = load_from_disk(f'./tokenized_dataset')
    model_config    :GridSearch = search(Gpt(), Rob())

    # TOKENIZER
    padding_side           :str = search('right', 'left')
    truncation_side        :str = search('right', 'left')

    # TRAINING HYPERPARAMETERS 
    batch_size                  = 16 # TinyStories uses 80, but I am training locally on my poor M1 Air
    num_train_epochs            = 1  # TinyStories doesn't mention
    gradient_accumulation_steps = 16 # TinyStories uses 16

    lr                   :float = search(8e-4, 1e-3, 2e-3)
    _train_steps                = len(dataset['train']) // (batch_size * gradient_accumulation_steps)
    eval_steps                  = _train_steps // 10 # evaluate every 10% of training steps

    # WANDB INFO
    project                     = 'baselines' 
    entity                      = 'tiny-transformers' 
    group                       = 'tok-exp'

    @property
    def tok(self):
        ''' properties are computed upon accessing them as attributes (`params.tok`) '''
        kwargs = {'padding_side': self.padding_side, 'truncation_side': self.truncation_side}
        if not hasattr(self, '__tok'):
            self.__tok = RobertaTokenizerFast.from_pretrained('10k-tok', **kwargs) \
                if isinstance(self.model_config, Rob) else \
                GPT2TokenizerFast.from_pretrained('10k-tok', **kwargs) 
        return self.__tok

    @property 
    def model(self): 
        if not hasattr(self, '__model'): 
            self.__model = RobertaForMaskedLM(RobertaConfig(**self.model_config.__dict__)) \
                if isinstance(self.model_config, Rob) else \
                GPTNeoForCausalLM(GPTNeoConfig(**self.model_config.__dict__))
        return self.__model

    @property 
    def output_dir(self) -> str:
        return os.path.join('models', self.group, self.model_name)

    @property
    def model_name(self) -> str: 
        name = '-'.join([
            'GPT' if isinstance(self.model, GPTNeoForCausalLM) else 'BERT',
            f'{self.model.num_parameters()//1e6:.1f}M',
            f'{self.model_config.num_layers if isinstance(self.model, GPTNeoForCausalLM) else self.model_config.num_hidden_layers}L', 
            f'{self.model_config.num_heads if isinstance(self.model, GPTNeoForCausalLM) else self.model_config.num_attention_heads}H', 
            f'{self.model_config.hidden_size}C',
            f'{self.model_config.intermediate_size}I',
            f'{self.lr}lr',
            # first letter of truncation_side 
            f'{self.truncation_side[0].upper()}T',
            # first letter of padding_side
            f'{self.padding_side[0].upper()}P'
        ])
        if TEST: name += '-TEST'
        return name 

    @property
    def trainer(self) -> Trainer: 

        training_args = TrainingArguments(

            seed       = 42,
            use_cpu    = False, # use GPU if available (not necessarily faster on laptops, but Apple's MPS have good support)

            output_dir = os.path.join(self.output_dir, 'checkpoints'),

            learning_rate               = self.lr,
            num_train_epochs            = self.num_train_epochs,
            per_device_train_batch_size = self.batch_size,
            per_device_eval_batch_size  = self.batch_size,
            gradient_accumulation_steps = self.gradient_accumulation_steps,

            evaluation_strategy = 'steps',
            eval_steps          = self.eval_steps if not TEST else 5,
            save_steps          = self.eval_steps if not TEST else 5,

            logging_first_step  = True,
            logging_steps       = self.eval_steps if not TEST else 1,
            report_to           = 'wandb' if not TEST else 'none',
        )

        model = self.model 
        tokenizer = self.tok 
        train_ds = self.dataset['train'] if not TEST else \
            self.dataset['train'].select(range(self.batch_size*10))
        eval_ds = self.dataset['validation'] if not TEST else \
            self.dataset['validation'].select(range(self.batch_size*4))

        trainer = Trainer(

            model               = model, 
            args                = training_args, 

            train_dataset       = train_ds,
            eval_dataset        = eval_ds,
            data_collator       = DataCollatorForLanguageModeling(
                tokenizer, mlm=isinstance(tokenizer, RobertaForMaskedLM)),
        )

        # print amount of training steps, and how often the model is evaluated
        print(f'''
        Retrieving Trainer for \033[1m{self.model_name}\033[0m ({model.num_parameters():,}M)

            Training for {self.num_train_epochs} epochs, {len(train_ds)} samples
            {self.batch_size} batch size, {self.gradient_accumulation_steps} accumulation steps.
            Evaluating every {self.eval_steps} steps, {len(eval_ds)} samples.
        ''')

        return trainer

def set_all_seeds(seed=42):

    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train(params: Hyperparams):

    # if output_dir exists, early exit if it contains a model.safetensors file 
    if os.path.exists(params.output_dir) and \
        os.path.exists(os.path.join(params.output_dir, 'model.safetensors')):
            print(f'\033[1m{params.model_name} has already been trained; skipping training\033[0m')
            return

    set_all_seeds()
    run_id = None
    if not DEBUG:
        run = wandb.init(
            entity=params.entity, project=params.project, 
            group=params.group, name=params.model_name, 
            config=params.__dict__)
        run_id = run.id

    else: print('\033[1mRUNNING IN DEBUG MODE \033[0m')

    trainer = params.trainer
    trainer.train()

    trainer.save_model(params.output_dir)

    # NOTE: this is where you *could* push your model to the hub;
    # or do that later after you are certain it is solid 
    # model.push_to_hub(save_dir)

    del trainer.model
    del trainer

    set_all_seeds()
    score = eval_and_aggregate(
        {'model': params.model_name, 'group': params.group, 'index': 0, 'no_train': False}
    )
    print(score)

    # we need to reinitialise as the babylm eval pipeline inits a whole bunch of runs
    wandb.init(entity=params.entity, project=params.project, id=run_id, resume='must')
    wandb.log(score)
    wandb.finish()

from tqdm.contrib.concurrent import process_map

if __name__ == '__main__':
    
    params = Hyperparams()
    print(params)

    # this will parallelise training across 4 CPUs
    # note you still need to deal with allocating GPUs
    # or fit multiple models on the same GPU. 
    # process_map(train, enumerate(params), max_workers=1)
    for param in params:
        print(param)
        train(param)


