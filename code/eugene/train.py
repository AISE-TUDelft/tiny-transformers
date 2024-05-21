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
        description="Train and evaluate specified model")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode quick test (limits max steps).")
    parser.add_argument('--gpu', type=int, required=True, help="Specify the GPU number to use.")
    parser.add_argument('--model_type', type=str, required=True, choices=['gpt', 'roberta'],
                        help="Specify the model type.")
    parser.add_argument('--sparsity_type', type=str, required=True, choices=['baseline', 'moe', 'cnt', 'pkm'],
                        help="Specify the sparsity type.")
    parser.add_argument('--sparsity_level', type=str, required=True, choices=['low', 'medium', 'high'],
                        help="Specify the sparsity level.")

    args = parser.parse_args()

    print(args)
