import os, sys, torch, wandb, numpy as np, random

# wandb.login()
TEST = len(sys.argv) > 0 and sys.argv[0] == 'test'

from datasets import load_from_disk 
from transformers import (
    RobertaForMaskedLM, RobertaConfig, RobertaTokenizerFast,
    GPTNeoForCausalLM, GPTNeoConfig, GPT2TokenizerFast,
    PreTrainedTokenizer, PretrainedConfig,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling,
    set_seed
)

from grid_search import GridSearch, search # cheeky library i wrote
from dataclasses import dataclass, field 

@dataclass 
class Gpt(GridSearch):
    ''' extend GridSearch with dataclass to search the values in `search`,
        can be used recursively. '''

    # EMBEDDING PARAMETERS
    vocab_size              = 10_000,                       # number of tokens in the vocabulary 
    hidden_size        :int = search([256, 384, 512]),      # embedding size (vector length) of each token 
    max_position_embeddings = 512,                          # maximum sequence length (context window)

    # BLOCKS (ATTN & FFN)
    num_layers         :int = search([2, 3]),               # number of transformer blocks
    attention_types         = [[["global", "local"], 1]],   # (GPT-Neo-specific) global and local attention 
    num_heads               = 4,                            # attention heads
    window_size             = 256,                          # (GPT-Neo-specific) for local attention 
    intermediate_size       = 1024,                         # size of 'up-projection' layer in FFN

    pad_token_id = 0,                   # need to specify this for tokenizer interop between models

    def instantiate(self): 
        return GPTNeoForCausalLM(GPTNeoConfig(**self.__dict__)), \
            GPT2TokenizerFast.from_pretrained('10k-tok')

@dataclass
class Rob(GridSearch):

    # EMBEDDING PARAMETERS
    vocab_size              = 10_000,
    hidden_size        :int = search([256, 384, 512]),
    # we add 1 as RoBERTa uses a special position eearbedding for the padding token (zero vector)
    max_position_embeddings = 512 + 1,

    # BLOCKS (of course naming is different in roberta :) )
    num_hidden_layers  :int = search([2,3]),
    num_attention_heads     = 4,
    intermediate_size       = 1024,

    pad_token_id = 0,

    def instantiate(self):
        return RobertaForMaskedLM(RobertaConfig(**self.__dict__)), \
            RobertaTokenizerFast.from_pretrained('10k-tok')

@dataclass 
class Hyperparams(GridSearch):

    dataset                     = load_from_disk(f'./tokenized_dataset')
    model           :GridSearch = search([g for g in Gpt()] + [r for r in Rob()])

    # TRAINING HYPERPARAMETERS 
    batch_size                  = 16 # TinyStories uses 80, but I am training locally on my poor M1 Air
    num_train_epochs            = 2  # TinyStories doesn't mention
    gradient_accumulation_steps = 16 # TinyStories uses 16

    lr                   :float = search([5e-4, 1e-3])
    _train_steps                = len(dataset) // (batch_size * gradient_accumulation_steps)
    eval_steps                  = _train_steps // 10 # evaluate every 10% of training steps

    # WANDB GROUP
    group                       = 'baseline' if not TEST else 'test'

    @property
    def model_name(self) -> str: 
        return '-'.join([
            'GPT' if isinstance(self.model, GPTNeoForCausalLM) else 'BERT',
            f'{self.model.num_parameters()//1e6:.1f}M',
            f'{self.config.num_layers if isinstance(self.model, GPTNeoForCausalLM) else self.config.num_hidden_layers}L', 
            f'{self.config.num_heads if isinstance(self.model, GPTNeoForCausalLM) else self.config.num_attention_heads}H', 
            f'{self.config.hidden_size}C',
            f'{self.config.intermediate_size}I',
            f'{self.lr}lr'
        ])

    @property 
    def output_dir(self) -> str:
        return os.path.join('models', self.group, self.model_name)

    @property
    def trainer(self) -> Trainer: 

        training_args = TrainingArguments(

            seed       = 42,
            use_cpu    = False, # use GPU if available (not necessarily faster on laptops, but Apple's MPS have good support)

            output_dir = os.path.join(self.output_dir, self.model_name),

            learning_rate               = self.lr,
            num_train_epochs            = self.num_train_epochs,
            per_device_train_batch_size = self.batch_size,
            per_device_eval_batch_size  = self.batch_size,
            gradient_accumulation_steps = self.gradient_accumulation_steps,

            evaluation_strategy = 'steps',
            eval_steps          = self.eval_steps if not TEST else 50,
            save_steps          = self.eval_steps if not TEST else 50,

            logging_first_step  = True,
            logging_steps       = 100 if not TEST else 25,
            report_to           = 'wandb',
        )
        model, tokenizer = self.model.instantiate()

        trainer = Trainer(

            model               = self.model, 
            args                = training_args, 

            train_dataset       = self.dataset['train'] if not TEST else self.dataset['train'][:100],
            eval_dataset        = self.dataset['eval']  if not TEST else self.datset['eval'][:100],
            data_collator       = DataCollatorForLanguageModeling(tokenizer, mlm=isinstance(model, RobertaForMaskedLM)),
        )

        # print amount of training steps, and how often the model is evaluated
        print(f'''
        Retrieving Trainer for \033[1m{self.model_name}\033[0m ({model.num_parameters():,}M)

            Training for {self.num_train_epochs} epochs, {len(self.dataset['train'])} samples
            {self.batch_size} batch size, {self.gradient_accumulation_steps} accumulation steps.
            Evaluating every {self.eval_steps} steps, {len(self.dataset['eval'])} samples.
        ''')

        return trainer

def set_all_seeds(seed=42):

    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train(params: Hyperparams, index=0):

    # this machine has 2 GPUs but it's faster to train models on a single GPU 
    torch.cuda.set_device(index)

    wandb.init(project='tiny-transformers', name=params.model_name, group=params.group, config=params.__dict__)
    trainer = params.trainer
    trainer.train()

    save_dir = os.path.join('models', params.group, params.model_name)
    trainer.save_model(save_dir)

    del trainer


from tqdm.contrib.concurrent import process_map

if __name__ == '__main__':
    
    for h in Hyperparams():
        print(h)

#         # set_all_seeds()
#         # train(h, i)
#         # set_all_seeds()
#         # evaluate(h, i)

