#!/bin/sh

for model_type in 'gpt' 'roberta'; do
    
    for sparsity_type in 'moe' 'cnt' 'pkm'; do
        # Inner loop: 'low', 'medium', and 'high'
        for sparsity_level in 'low' 'medium' 'high'; do
            sbatch delftblue_pretrain.sh false $model_type $sparsity_type $sparsity_level
        done
    done
done

sbatch delftblue_pretrain.sh false gpt baseline low
sbatch delftblue_pretrain.sh false roberta baseline low