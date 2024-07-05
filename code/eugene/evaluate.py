import json
import os
import transformers
import torch
import numpy as np


from common.eval_baselines import eval_and_aggregate

import wandb 

import argparse

# set seed
def set_seeds(seed=123):
    np.random.seed(seed)
    transformers.set_seed(seed)
    torch.cuda.manual_seed_all(seed)

# training loop
def get_model_name(model_type, sparsity_type, sparsity_level):
    if sparsity_type == 'baseline':
        return f"{model_type}_{sparsity_type}"
    else:
        return f"{model_type}_{sparsity_type}_{sparsity_level}"


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
    model_name = get_model_name(args.model_type, args.sparsity_type, args.sparsity_level)
    print(eval_and_aggregate(f"../../models/eugene/{model_name}"))