import os, torch, wandb, numpy as np, random, datetime as dt, time
from datasets import load_from_disk, load_dataset
from pprint import pprint
from safetensors.torch import load_model

from transformers import (
    RobertaTokenizerFast, GPT2TokenizerFast, 
    Trainer, TrainingArguments, DataCollatorForLanguageModeling,
    set_seed
)

from common.eval_baselines import eval_and_aggregate
from common.grid_search import GridSearch
from dataclasses import dataclass

def set_all_seeds(seed=42):

    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_dataset(context_length=512, debug=False):

    num_proc = 5 if debug else 64
    dataset_path = f'./tokenized_dataset/{context_length}'

    if not os.path.exists(dataset_path):
        tok_gpt = GPT2TokenizerFast.from_pretrained('common/10k-tok')
        tok_gpt.model_max_length = context_length

        dataset = load_dataset('roneneldan/tinystories', num_proc=16)
        dataset = dataset.map(
            lambda x: tok_gpt(x['text'], truncation=True, padding='max_length'),
            batched=True, num_proc=num_proc, batch_size=1_000)                 # change num_proc to 1 if multithread issues

        dataset.save_to_disk(dataset_path, num_proc=5)


    dataset = load_from_disk(dataset_path, keep_in_memory=True)
    if debug: # only use 1% of the dataset 
        dataset['train'] = dataset['train'].select(range(len(dataset['train'])//100))
        dataset['validation'] = dataset['validation'].select(range(len(dataset['validation'])//100))

    return dataset


@dataclass 
class Hyperparams(GridSearch):

    model_config         :GridSearch 

    # WANDB INFO
    group                       :str 
    project                     :str = 'aral' 
    entity                      :str = 'tiny-transformers' 

    debug                      :bool = False

    # default hyperparams
    batch_size                  :int = 16 # TinyStories uses 80, but I am training locally on my poor M1 Air
    num_train_epochs            :int = 1  # TinyStories doesn't mention
    gradient_accumulation_steps :int = 16 # TinyStories uses 16
    lr                        :float = 1e-3
    eval_steps                :float = 0.1 / num_train_epochs

    def get_dataset(self): 
        return get_dataset(self.model_config.max_position_embeddings, debug=self.debug)

    @property 
    def output_dir(self) -> str:
        return os.path.join('models', self.project, self.group, self.model_name)

    @property
    def model(self):
        if not hasattr(self, '_model'):
            self._model = self.model_config.create_model()
            checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
            if os.path.exists(checkpoint_dir):

                # NOTE: we also reduce training epochs proportionally when loading the trainer
                last_checkpoint = os.listdir(checkpoint_dir)[-1]
                print(f'Loading from last checkpoint: {last_checkpoint}')
                self._model = self._model.from_pretrained(os.path.join(checkpoint_dir, last_checkpoint))
                self.last_checkpoint = last_checkpoint
        return self._model

    @property
    def model_name(self) -> str: 
        return '-'.join([
            'GPT',
            f'{self.model.num_parameters()//1e6:.1f}M',
            f'{self.model_config.num_layers if "GPT" in self.model_config.__class__.__name__ else self.model_config.num_hidden_layers}L', 
            f'{self.model_config.num_heads if "GPT" in self.model_config.__class__.__name__ else self.model_config.num_attention_heads}H', 
            f'{self.model_config.hidden_size}C',
            f'{self.model_config.intermediate_size}I',
            f'{self.lr}lr',
            f'{self.num_train_epochs}e',
            *self.model_config.model_name
        ]) 

    @property
    def tokenizer(self):
        if not hasattr(self, '_tokenizer'):
            self._tokenizer = self.model_config.create_tokenizer()
        return self._tokenizer

    @property
    def trainer(self) -> Trainer: 

        model = self.model
        tokenizer = self.tokenizer

        dataset = self.get_dataset()
        train_ds, eval_ds = dataset['train'], dataset['validation']

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
            eval_steps          = self.eval_steps,
            save_steps          = self.eval_steps,

            logging_first_step  = True,
            logging_steps       = self.eval_steps / 100,
            report_to           = 'wandb' if not self.debug else 'none',
        )
        trainer = Trainer(

            model               = model, 
            args                = training_args, 

            train_dataset       = train_ds,
            eval_dataset        = eval_ds,
            data_collator       = DataCollatorForLanguageModeling(
                tokenizer, mlm=isinstance(tokenizer, RobertaTokenizerFast)),
        )

        # print amount of training steps, and how often the model is evaluated
        print(f'''
    Retrieving Trainer for \033[1m{self.model_name}\033[0m ({model.num_parameters():,}M)
    Saving to {training_args.output_dir}. training on GPU: {torch.cuda.is_available()}

        Training for {self.num_train_epochs} epochs, {len(train_ds)} samples
        {self.batch_size} batch size, {self.gradient_accumulation_steps} accumulation steps.
        Evaluating every {self.eval_steps} steps, {len(eval_ds)} samples.
    
        ''')

        self.trainer = trainer 
        return trainer

    def should_continue_wandb(self) -> bool: 
        ''' set up wandb run (adds self.run_id), return true if the user wants to continue ''' 
        
        if self.debug: return False

        api = wandb.Api()
        runs = sorted(api.runs(path=f'{self.entity}/{self.project}'), 
                      key = lambda run: dt.datetime.fromisoformat(run.created_at))

        # If there exists a run on wandb with the same name
        if not any([self.model_name == run.name for run in runs]): 
            for run in runs: 
                if run.name == self.model_name: break 

            # ask the user if they want to continue that run
            resume = input('Found run \033[1m{run.id}\033[0m on wandb. Continue? [y/n] ')
            if resume.lower() == 'y': 

                # resume that run and return its id 
                run = wandb.init(entity = self.entity, project = self.project, 
                                 id = run.id, resume = 'must')
                self.run_id = run.id 
                return True 

            # otherwise, give the user some time to undo if they are wrong
            time.sleep(3)
 
        # initialise a new wandb run
        run = wandb.init(entity=self.entity, project=self.project, 
                         group=self.group, name=self.model_name, 
                         config=self.__dict__)
        self.run_id = run.id 
        return False

    def train(self):
        ''' train this hyperparam combination, optionally from a checkpoint ''' 

        print(f'\t\033[1m> Training {self.model_name}\033[0m')
        # Skip if model weights are present 
        if os.path.exists(self.output_dir) and \
                os.path.exists(os.path.join(self.output_dir, 'model.safetensors')):
            print(f'\t{self.model_name} has already been trained; skipping training')
            return 

        # do the actual training
        set_all_seeds()
        trainer = self.trainer
        trainer.train(resume_from_checkpoint=self.should_continue_wandb())
        trainer.save_model(self.output_dir)

        # cleanup
        wandb.finish()
        del trainer.model
        del trainer

        # NOTE: this is where you *could* push your model to the hub;
        # or do that later after you are certain it is solid 
        # model.push_to_hub(save_dir)


    def evaluate(self): 
        ''' evaluate on babylm, returning score dict and wandb id ''' 

        print(f'\t\033[1m> Evaluating {self.model_name}\033[0m')
        set_all_seeds()
        score = eval_and_aggregate(self.output_dir, 0, debug=self.debug)

        # Try to resume the wandb run
        if self.should_continue_wandb():
            wandb.log(score)
            wandb.finish()

        # otherwise just print to console
        else:
            print(f'Could not find a run on wandb to save the scores to!!')
            print(score)

        return score

    def train_and_eval(self, debug=False):

        if self.debug: print('\033[1mRUNNING IN DEBUG MODE \033[0m')

        self.train()
        return self.evaluate()


