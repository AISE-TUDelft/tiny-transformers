#!/bin/bash

# take in one argument from command line, corresponding 
# to the model directory
model_dir=$1

if [[ $model_dir == *"GPT"* ]]; then
    model_type="decoder"
else
    model_type="encoder"
fi


printf "\n\n\n\033[1mRunning $model_type on BLiMP for \n$model_dir\033[0m\n\n"
echo $model_dir $model_type
python babylm_eval.py $model_dir $model_type

printf "\n\n\033[1mRunning (Super)GLUE for \n$model_dir\033[0m\n\n"
./finetune_all_tasks.sh $model_dir
