import os
import transformers
from sparse_gpt_neo import SparseGPTNeoForCausalLM
from configuration_sparse_gpt_neo import SparseGPTNeoConfig
from sparse_roberta import SparseRobertaForMaskedLM
from configuration_sparse_roberta import SparseRobertaConfig
from sparse_ffn.sparsity_types import SparsityType
import wandb

from code.common.eval_baselines import eval_and_aggregate

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that takes a debug flag, a GPU number, a model type, and a sparsity type.")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode for detailed logging.")
    parser.add_argument('--gpu_number', type=int, required=True, help="Specify the GPU number to use.")
    parser.add_argument('--model_type', type=str, required=True, choices=['type1', 'type2', 'type3'],
                        help="Specify the model type.")
    parser.add_argument('--sparsity_type', type=str, required=True, choices=['sparse', 'dense'],
                        help="Specify the sparsity type.")

    args = parser.parse_args()

    print(args)
