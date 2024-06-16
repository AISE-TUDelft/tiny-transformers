#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

# import torch
# from torch import nn
from transformers import GPTNeoForCausalLM, GPTNeoConfig
import json
from typing import Dict, Any
# import sys
# import importlib.util
# # Define the absolute path to the trainer.py file
# trainer_file_path = '/opt/conda/lib/python3.11/site-packages/transformers/trainer.py'

# # Load the module
# spec = importlib.util.spec_from_file_location("trainer", trainer_file_path)
# trainer_module = importlib.util.module_from_spec(spec)
# sys.modules["trainer"] = trainer_module
# spec.loader.exec_module(trainer_module)

class CustomGPTNeoConfig(GPTNeoConfig):
    model_type = "custom-gqa-gpt-neo"
    def __init__(self, num_kv_heads=None, kqv_size=None, **kwargs):
        if num_kv_heads is None:
            self.num_kv_heads = kwargs['num_heads']
        else:
            self.num_kv_heads = num_kv_heads
        if kqv_size is None:
            self.kqv_size = kwargs['hidden_size']
        else:
            self.kqv_size = kqv_size
        super().__init__(**kwargs)
    def to_dict(self):
        # Get the dictionary representation of the base class
        config_dict = super().to_dict()
        
        # Add custom attributes
        config_dict['num_kv_heads'] = self.num_kv_heads
        config_dict['kqv_size'] = self.kqv_size
        
        return config_dict
    
    def to_diff_dict(self) -> Dict[str, Any]:
        """
        Removes all attributes from config which correspond to the default config attributes for better readability and
        serializes to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        # Get the dictionary representation of this instance
        config_dict = self.to_dict()

        # Get the default config dictionary
        default_config_dict = PretrainedConfig().to_dict()

        # Get the class-specific config dictionary if applicable
        class_config_dict = self.to_dict() if not self.is_composition else {}

        # Initialize the serializable config dictionary
        serializable_config_dict = {}

        # Only serialize values that differ from the default config
        for key, value in config_dict.items():
            if (
                isinstance(getattr(self, key, None), PretrainedConfig)
                and key in class_config_dict
                and isinstance(class_config_dict[key], dict)
            ):
                # For nested configs, clean the diff recursively
                diff = super.recursive_diff_dict(value, class_config_dict[key], config_obj=getattr(self, key, None))
                if "model_type" in value:
                    # Needs to be set even if it's not in the diff
                    diff["model_type"] = value["model_type"]
                if len(diff) > 0:
                    serializable_config_dict[key] = diff
            elif (
                key not in default_config_dict
                or key == "transformers_version"
                or value != default_config_dict[key]
                or (key in class_config_dict and value != class_config_dict[key])
            ):
                serializable_config_dict[key] = value

        if hasattr(self, "quantization_config"):
            serializable_config_dict["quantization_config"] = (
                self.quantization_config.to_dict()
                if not isinstance(self.quantization_config, dict)
                else self.quantization_config
            )

            # Pop the `_pre_quantization_dtype` as torch.dtypes are not serializable.
            _ = serializable_config_dict.pop("_pre_quantization_dtype", None)

        # Ensure custom attributes are included if they differ from the default
        for custom_attr in ['num_kv_heads', 'kqv_size']:
            if hasattr(self, custom_attr) and (custom_attr not in default_config_dict or getattr(self, custom_attr) != default_config_dict.get(custom_attr)):
                serializable_config_dict[custom_attr] = getattr(self, custom_attr)

        self.dict_torch_dtype_to_str(serializable_config_dict)

        if "_attn_implementation_internal" in serializable_config_dict:
            del serializable_config_dict["_attn_implementation_internal"]

        return serializable_config_dict
    
    def to_json_string(self, use_diff: bool = True) -> str:
        """
        Serializes this instance to a JSON string.

        Args:
            use_diff (`bool`, *optional*, defaults to `True`):
                If set to `True`, only the difference between the config instance and the default `PretrainedConfig()`
                is serialized to JSON string.

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        if use_diff is True:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

import torch
from torch import nn
from transformers import GPTNeoForCausalLM, GPTNeoForSequenceClassification, AutoModelForSequenceClassification, AutoModelForCausalLM, AutoConfig
from transformers.utils import logging
from typing import Optional
# from transformers.models.auto.configuration_auto import CONFIG_MAPPING
# from transformers.models.auto.modeling_auto import MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING

logger = logging.get_logger(__name__)

class GPTNeoGQASelfAttention(nn.Module):
        def __init__(self, config, attention_type):
            super().__init__()
            # print("KQV_size: ", kqv_size)
            # print("Got to constructor")
            self.config = config

            self.num_kv_heads = config.num_kv_heads
            self.gqa_factor = config.num_kv_heads / self.config.num_heads

            self.kqv_size = config.kqv_size

            max_positions = config.max_position_embeddings
            bias = torch.tril(torch.ones((max_positions, max_positions), dtype=bool)).view(
                1, 1, max_positions, max_positions
            )

            # local causal self attention is a sliding window where each token can only attend to the previous
            # window_size tokens. This is implemented by updating the causal mask such that for each token
            # all other tokens are masked except the previous window_size tokens.
            if attention_type == "local":
                bias = torch.bitwise_xor(bias, torch.tril(bias, -config.window_size))

            self.register_buffer("bias", bias, persistent=False)
            self.register_buffer("masked_bias", torch.tensor(-1e9), persistent=False)

            self.attn_dropout = nn.Dropout(float(config.attention_dropout))
            self.resid_dropout = nn.Dropout(float(config.resid_dropout))
            self.is_causal = True

            self.embed_dim = config.hidden_size
            self.num_heads = config.num_heads
            self.head_dim = self.kqv_size // self.num_heads
            # print("Left: ", self.head_dim * self.num_heads)
            # print("Right: ", self.kqv_size)
            if self.head_dim * self.num_heads != self.kqv_size:
                raise ValueError(
                    f"embed_dim and kqv size must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                    f" {self.num_heads})."
                )
            if self.kqv_size * self.gqa_factor != int(self.kqv_size * self.gqa_factor):
                raise ValueError (
                    f"kqv size must be divisible by number of kv heads (got `kqv_size`: {self.kqv_size} and `num_kv_heads`:"
                    f" {self.num_kv_heads})."
                )

            self.k_proj = nn.Linear(self.embed_dim, int(self.kqv_size * self.gqa_factor), bias=False)
            self.v_proj = nn.Linear(self.embed_dim, int(self.kqv_size * self.gqa_factor), bias=False)

            self.q_proj = nn.Linear(self.embed_dim, self.kqv_size, bias=False)
            self.out_proj = nn.Linear(self.kqv_size, self.embed_dim, bias=True)

        def _split_heads(self, tensor, num_heads, attn_head_size):
            """
            Splits hidden_size dim into attn_head_size and num_heads
            """
            # attn_head_size is self.head_dim = self.embed_dim // self.num_heads
            # print("Size in split heads passed as the input: ", tensor.shape)
            new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
            # print("New shape in split heads before permuting: ", new_shape)
            tensor = tensor.view(new_shape)
            return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

        def _merge_heads(self, tensor, num_heads, attn_head_size):
            """
            Merges attn_head_size dim and num_attn_heads dim into hidden_size
            """
            tensor = tensor.permute(0, 2, 1, 3).contiguous()
            new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
            return tensor.view(new_shape)

        def _attn(self, query, key, value, attention_mask=None, head_mask=None):
            # calculates attention outputs and weights for one query
            # Keep the attention weights computation in fp32 to avoid overflow issues
            query = query.to(torch.float32)
            key = key.to(torch.float32)
            # print("_attn key shape: ", key.shape)
            # print("_attn query shape: ", query.shape)
            # transpose part is not changed, but dot product has to be edited
            # attn_weights = torch.matmul(query, key.transpose(-1, -2))
            # print("_attn weights torch.matmul(query, key.transpose(-1, -2)): ", attn_weights.shape)

            # Transpose the key tensor along the last two dimensions
            key_transposed = key.transpose(-1, -2)
            # Initialize a list to collect the attention weights for each head
            attn_weights_list = []

            # Loop over each head (second dimension)
            i, j = 0, 0
            query_leftover = self.num_heads % self.num_kv_heads
            queries_per_group = (self.num_heads // self.num_kv_heads)
            while i < key.shape[1] :
                key_head = key_transposed[:, i, :, :]
                if query_leftover <= 0:
                    toGroup = queries_per_group
                else:
                    toGroup = queries_per_group + 1
                    query_leftover -= 1
                for k in range (j, j + toGroup):
                    query_head = query[:, k, :, :]
                    attn_weight_qk = torch.matmul(query_head, key_head)
                    attn_weights_list.append(attn_weight_qk)
                j += toGroup
                i += 1

            attn_weights_decomposed = torch.stack(attn_weights_list, dim=1)

            query_length, key_length = query.size(-2), key.size(-2)     # no need to change
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]

            mask_value = torch.finfo(attn_weights_decomposed.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.tensor(mask_value, dtype=attn_weights_decomposed.dtype).to(attn_weights_decomposed.device)
            attn_weights_decomposed = torch.where(causal_mask, attn_weights_decomposed, mask_value)

            if attention_mask is not None:
                # Apply the attention mask
                attn_weights_decomposed = attn_weights_decomposed + attention_mask

            attn_weights_decomposed = nn.functional.softmax(attn_weights_decomposed, dim=-1)
            attn_weights_decomposed = attn_weights_decomposed.to(value.dtype)
            attn_weights_decomposed = self.attn_dropout(attn_weights_decomposed)

            # Mask heads if we want to
            if head_mask is not None:
                attn_weights_decomposed = attn_weights_decomposed * head_mask

            # assert torch.allclose(attn_weights_decomposed, attn_weights)
            # Initialize a list to collect the attention outputs for each head
            attn_output_list = []
            # Loop over each query head and apply the corresponding attention weights to the value heads
            i, j = 0, 0
            query_leftover = self.num_heads % self.num_kv_heads
            while i < value.shape[1]:
                value_head = value[:, i, :, :]
                if query_leftover <= 0:
                    toGroup = queries_per_group
                else:
                    toGroup = queries_per_group + 1
                    query_leftover -= 1
                for k in range(j, j + toGroup):
                    attn_weight_head = attn_weights_decomposed[:, k, :, :]
                    attn_output_head = torch.matmul(attn_weight_head, value_head)
                    attn_output_list.append(attn_output_head)
                j += toGroup
                i += 1
            attn_output_decomposed = torch.stack(attn_output_list, dim=1)  # Shape: [1, 4, 12, 16]

            return attn_output_decomposed, attn_weights_decomposed

        def forward(
            self,
            hidden_states,
            attention_mask=None,
            layer_past=None,
            head_mask=None,
            use_cache=False,
            output_attentions=False,
        ):
            # use_cache is set to true
            # output_attentions is set to false
            # print("Got to forward")
            # print("Hidden state size: ", hidden_states.size())
            query = self.q_proj(hidden_states)
            # print("Query size: ", query.size())
            key = self.k_proj(hidden_states)
            value = self.v_proj(hidden_states)
            # print("key shape: ", key.shape)
            # print("value shape: ", value.shape)

            query = self._split_heads(query, self.num_heads, self.head_dim)
            key = self._split_heads(key, self.num_kv_heads, self.head_dim)
            value = self._split_heads(value, self.num_kv_heads, self.head_dim)

            if layer_past is not None:
                # since cache is used, previous keys and values are extracted here
                # each time, the dimensionality is increased by 1 along sequence length dimension
                past_key = layer_past[0]
                past_value = layer_past[1]
                # print("past key shape: ", past_key.shape)
                # print("past value shape: ", past_value.shape)
                # print("key shape: ", key.shape)
                # print("value shape: ", value.shape)
                key = torch.cat((past_key, key), dim=-2)
                value = torch.cat((past_value, value), dim=-2)
                # print("key after: ", key.shape)
                # print("value after: ", value.shape)

            if use_cache is True:
                present = (key, value)
            else:
                present = None
            # print("Key shape before calculating: ", key.shape)
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
            # print("Grid shape: ", attn_output.shape)
            attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
            # print("Shape after merging heads: ", attn_output.shape)
            attn_output = self.out_proj(attn_output)
            attn_output = self.resid_dropout(attn_output)

            # print("Final result: ", attn_output.shape)

            outputs = (attn_output, present)
            if output_attentions:
                outputs += (attn_weights,)
            # print("Final result 2: ", outputs[0].shape)
            return outputs  # a, present, (attentions)

class CustomGPTNeoForCausalLM(GPTNeoForCausalLM):
    config_class = CustomGPTNeoConfig
    def __init__(self, config: CustomGPTNeoConfig):
        self.config = config
        super().__init__(config)
        self.kqv_size = config.kqv_size
        self.num_kv_heads = config.num_kv_heads
        for block in self.transformer.h:
            block.attn.attention = GPTNeoGQASelfAttention(self.config, block.attn.attention_type)

class CustomGPTNeoForSequenceClassification(GPTNeoForSequenceClassification):
    config_class = CustomGPTNeoConfig
    def __init__(self, config: CustomGPTNeoConfig):
        super().__init__(config)
        self.config = config
        self.kqv_size = config.kqv_size
        self.num_kv_heads = config.num_kv_heads
        
        # Replace the attention mechanism in each transformer block
        for block in self.transformer.h:
            block.attn.attention = GPTNeoGQASelfAttention(self.config, block.attn.attention_type)


import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
import evaluate
import re
import pdb

import datasets
import numpy as np
from datasets import load_dataset
from sklearn.metrics import f1_score, matthews_corrcoef

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    EvalPrediction,
    HfArgumentParser,
    IntervalStrategy,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.27.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    patience: Optional[int] = field(
       default=None,
       metadata={
            "help": (
                "Number of evaluation steps without improvement > epsilon before stopping fine-tuning. "
                "Requires the use of the --eval_every argument."
            )
       },
    )
    eval_every: Optional[int] = field(
        default=None,
        metadata = {
            "help": (
                "Number of steps between evaluations (MUST be set if patience is set)."
            )
        }
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    freeze_model: bool = field(
        default=False,
        metadata={"help": "Whether to freeze the parameters of the base model."}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Check for the use of early stopping
    if data_args.patience: 
        training_args.eval_steps = data_args.eval_every
        training_args.save_total_limit = 1
        training_args.load_best_model_at_end = True
        training_args.evaluation_strategy = "steps"
        callbacks = [EarlyStoppingCallback(early_stopping_patience=data_args.patience,
                                           early_stopping_threshold=0.001)]
    else:
        callbacks = None

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_glue", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            "glue",
            data_args.task_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

        # Get the test dataset: you can provide your own CSV/JSON test file (see below)
        # when you use `do_predict` without specifying a GLUE benchmark task.
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert (
                    test_extension == train_extension
                ), "`test_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                train_extension = data_args.train_file.split(".")[-1]
                validation_extension = data_args.validation_file.split(".")[-1]
                assert (
                    validation_extension == train_extension
                ), "`validation_file` should have the same extension (csv or json) as `train_file`."
                data_files["test"] = data_args.validation_file

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if data_args.train_file.endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset(
                "csv",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        else:
            # Loading a dataset from local json files
            raw_datasets = load_dataset(
                "json",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = raw_datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = raw_datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)
    
    is_binary = (num_labels == 2)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    # config = AutoConfig.from_pretrained(
    #     model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    #     num_labels=num_labels,
    #     finetuning_task=data_args.task_name,
    #     cache_dir=model_args.cache_dir,
    #     revision=model_args.model_revision,
    #     use_auth_token=True if model_args.use_auth_token else None,
    # )
    # print("Config name: ", model_args.config_name)
    # print("Model name: ", model_args.model_name_or_path)
    # print("Config type: ", type(config))

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    try:
        # model = AutoModelForSequenceClassification.from_pretrained(
        #     model_args.model_name_or_path,
        #     from_tf=bool(".ckpt" in model_args.model_name_or_path),
        #     config=config,
        #     cache_dir=model_args.cache_dir,
        #     revision=model_args.model_revision,
        #     use_auth_token=True if model_args.use_auth_token else None,
        #     ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        # )
        model_name = model_args.model_name_or_path
        pattern_kv = r'(\d+\.\d+)KV'  # Matches decimal numbers followed by 'KV'
        pattern_h = r'(\d+)H'  # Matches numbers followed by 'H'
        pattern_f = r'(\d+)F'  # Matches numbers followed by 'F'
        pattern_c = r'(\d+)C'  # Matches numbers followed by 'C'

        # Extract values using regex
        match_kv = re.search(pattern_kv, model_name)
        match_h = re.search(pattern_h, model_name)
        match_f = re.search(pattern_f, model_name)
        match_c = re.search(pattern_c, model_name)

        if match_kv is None:
            group_factor = 1
        else:
            group_factor = float(match_kv.group(1))
        num_heads = int(match_h.group(1))
        kqv_factor = float(match_f.group(1))
        embedding_size = int(match_c.group(1))

        num_kv_heads = int(num_heads * group_factor)
        kqv_size = int(embedding_size // kqv_factor)

        config_neo = GPTNeoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task=data_args.task_name,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        config_dict_1 = config_neo.to_dict()
        # Check if 'kqv_size' is present in config_dict and remove it if it is
        if 'kqv_size' in config_dict_1:
            del config_dict_1['kqv_size']
        # Check if 'num_kv_heads' is present in config_dict and remove it if it is
        if 'num_kv_heads' in config_dict_1:
            del config_dict_1['num_kv_heads']
        config = CustomGPTNeoConfig(num_kv_heads=num_kv_heads, kqv_size=kqv_size, **config_dict_1)

        AutoConfig.register("custom-gqa-gpt-neo", CustomGPTNeoConfig)  
        AutoModelForSequenceClassification.register(CustomGPTNeoConfig, CustomGPTNeoForSequenceClassification)
        AutoModelForCausalLM.register(CustomGPTNeoConfig, CustomGPTNeoForCausalLM)
        model_1 = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes
        )
        # print(type(model_1))
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes
        )

    except ValueError as e:
        from transformers import T5Config
        if isinstance(config, T5Config):
            from transformers_modified.t5 import T5ForSequenceClassification
            model = T5ForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
            )
        else:
            raise e
    
    # Freeze all parameters (including embeddings) except the classifier head
    if model_args.freeze_model:
        for name, param in model.named_parameters():
            if "classifier" not in name and not name.startswith("score"): # classifier layer
                param.requires_grad = False

    # Preprocessing the raw_datasets
    template = None
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name not in ("label", "idx")]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        elif "question1" in non_label_column_names and "question2" in non_label_column_names:
            sentence1_key, sentence2_key = "question1", "question2"
        elif "premise" in non_label_column_names and "hypothesis" in non_label_column_names:
            sentence1_key, sentence2_key = "premise", "hypothesis"
        elif "question" in non_label_column_names and "passage" in non_label_column_names:
            sentence1_key, sentence2_key = "question", "passage"
        elif "question" in non_label_column_names and "sentence" in non_label_column_names:
            sentence1_key, sentence2_key = "question", "sentence"
        # special cases
        elif "paragraph" in non_label_column_names and \
             "question" in non_label_column_names and \
             "answer" in non_label_column_names:        # MultiRC
            sentence1_key, sentence2_key = ["question", "answer"], "paragraph"
            template = "Question: {} Answer: {}"
        elif "text" in non_label_column_names and \
             "span1_text" in non_label_column_names and \
             "span2_text" in non_label_column_names:    # WSC
            sentence1_key, sentence2_key = ["span2_text", "span1_text"], "text"
            template = "Does \"{}\" refer to \"{}\" in this passage?"
        elif "sentence" in non_label_column_names and "linguistic_feature_type" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence", None
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        if isinstance(sentence1_key, list):
            keya, keyb = examples[sentence1_key[0]], examples[sentence1_key[1]]
            keys1 = [template.format(ka, kb) for ka, kb in zip(keya, keyb)]
            args = (
                (keys1, examples[sentence2_key])
            )
        else:
            args = (
                (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
            )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict or data_args.task_name is not None or data_args.test_file is not None:
        if "test" not in raw_datasets and "test_matched" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test_matched" if data_args.task_name == "mnli" else "test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = evaluate.load("glue", data_args.task_name)
    else:
        metric = evaluate.load("accuracy")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        elif is_binary:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item(),
                    "f1": f1_score(y_true=p.label_ids, y_pred=preds, average="binary"),
                    "mcc": matthews_corrcoef(y_true=p.label_ids, y_pred=preds)}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None
    
    if is_regression:
        training_args.metric_for_best_model = "mse"
    elif is_binary:
        training_args.metric_for_best_model = "f1"
        training_args.greater_is_better = True
    else:
        training_args.metric_for_best_model = "accuracy"
        training_args.greater_is_better = True
    
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks = callbacks,
    )
    # print("Trainer model: ", type(trainer.model))
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        # pdb.set_trace()
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            valid_mm_dataset = raw_datasets["validation_mismatched"]
            if data_args.max_eval_samples is not None:
                max_eval_samples = min(len(valid_mm_dataset), data_args.max_eval_samples)
                valid_mm_dataset = valid_mm_dataset.select(range(max_eval_samples))
            eval_datasets.append(valid_mm_dataset)
            combined = {}

        for eval_dataset, task in zip(eval_datasets, tasks):
            if "condition" in non_label_column_names:
                eval_dataset = eval_dataset.filter(lambda example: example["condition"] == "test")
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

            if task == "mnli-mm":
                metrics = {k + "_mm": v for k, v in metrics.items()}
            if task is not None and "mnli" in task:
                combined.update(metrics)

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", combined if task is not None and "mnli" in task else metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        # We do not use MNLI test data.
        tasks = [data_args.task_name]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            predict_datasets.append(raw_datasets["test_mismatched"])

        # Removing the `label` columns because it contains -1 and Trainer won't like that.
        predict_dataset = predict_dataset.remove_columns("label")
        predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
        predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

        output_predict_file = os.path.join(training_args.output_dir, f"predict_results.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                logger.info(f"***** Predict results *****")
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    if is_regression:
                        writer.write(f"{index}\t{item:3.3f}\n")
                    else:
                        item = label_list[item]
                        writer.write(f"{index}\t{item}\n")

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
    if data_args.task_name is not None:
        kwargs["language"] = "en"
        kwargs["dataset_tags"] = "glue"
        kwargs["dataset_args"] = data_args.task_name
        kwargs["dataset"] = f"GLUE {data_args.task_name.upper()}"

    """
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)
    """

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
