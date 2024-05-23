#!/bin/bash

# take in one argument from command line, corresponding 
# to the model directory
model_dir=$1
tok_files='10k-tok/*'

# we assume you have an environment named `babylm` 
# with all the packages, as described in `evaluate.ipynb`
conda activate babylm 

# copy the tokenizer to model_dir
cp $tok_files $model_dir

cd evaluation-pipeline


# store model_type as decoder if model_dir contains GPT 
# else it's an encoder 
if [[ $model_dir == *"GPT"* ]]; then
    model_type="decoder"
else
    model_type="encoder"
fi


printf "\n\n\n\033[1mRunning $model_type on BLiMP for \n$model_dir\033[0m\n\n"
python babylm_eval.py $model_dir $model_type

printf "\n\n\033[1mRunning (Super)GLUE for \n$model_dir\033[0m\n\n"
./finetune_all_tasks.sh $model_dir

# python collect_results.py $model_dir
