#!/bin/sh

for model_type in 'gpt' 'roberta'; do
    
    for sparsity_type in 'baseline' 'moe'; do
        # Inner loop: 'low', 'medium', and 'high'
        for sparsity_level in 'low' 'medium'; do
            sbatch delfblue_pretrain.sh true $model_type $sparsity_type $sparsity_level
        done
    done
done