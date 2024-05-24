#!/bin/bash

# NOTE: make sure you're in an environment with babylm dependencies installed.

# take in one argument from command line, corresponding 
# to the model directory
echo "Root directory: $(pwd)"
model_dir=$1
tok_files='10k-tok/*'

# Resolve the absolute path of the model_dir
model_dir=$(cd "$model_dir" && pwd)

# check if model_dir is a valid directory
if [ ! -d "$model_dir" ]; then
    echo "Error: model directory '$model_dir' does not exist."
    exit 1
fi

# copy the tokenizer to model_dir
cp $tok_files $model_dir

cd evaluation-pipeline

echo "Current directory: $(pwd)"

# echo the model dir
echo "Model directory: $model_dir"

# check if config.json exists in model_dir
if [ ! -f "$model_dir/config.json" ]; then
    echo "Error: config.json not found in model directory '$model_dir'."
    exit 1
fi



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

python collect_results.py $model_dir
