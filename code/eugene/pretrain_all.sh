#!/bin/bash

source venv/Scripts/activate

# Check for the correct number of arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <gpu> <debug>"
    exit 1
fi

# Assign arguments to variables
GPU=$1
DEBUG=$2


# Check if GPU argument is an integer
if ! [[ "$GPU" =~ ^[0-9]+$ ]]; then
    echo "Error: GPU must be an integer."
    exit 1
fi


for model_type in 'gpt' 'roberta'; do
    
    for sparsity_type in 'baseline' 'moe' 'cnt' 'pkm'; do
        # Inner loop: 'low', 'medium', and 'high'
        for sparsity_level in 'low' 'medium' 'high'; do
            if [ "$DEBUG" = "true" ]; then
                python pretrain.py --debug --model_type $model_type --gpu $GPU --sparsity_type $sparsity_type --sparsity_level $sparsity_level
            else
                python pretrain.py --model_type $model_type --gpu $GPU --sparsity_type $sparsity_type --sparsity_level $sparsity_level
            fi
        done
    done
done