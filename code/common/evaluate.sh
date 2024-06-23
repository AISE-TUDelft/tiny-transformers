#!/bin/bash

# NOTE: make sure you're in an environment with babylm dependencies installed.
echo $CWD $(pwd) 

# take in one argument from command line, corresponding 
# to the model directory
model_dir=$1
tok_files='10k-tok/*'
max_seq_length=512

if [ "$2" == "debug" ]; then
	max_epochs=1
else 
	max_epochs=10 # default value in finetune_all_tasks.sh
fi 

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


## BLIMP (note the batch size is defined in two places)
printf "\n\n\n\033[1mRunning BLiMP ($model_type) for \n$model_dir\n"
printf "max_seq_length=$max_seq_length \tmax_epochs=$max_epochs \tbatch_size=80\033[0m\n"
# args: dir, type, batch_size, 
python babylm_eval.py $model_dir $model_type 80

## GLUE (note the batch size is defined in two places)
printf "\n\n\033[1mRunning (Super)GLUE for \n$model_dir\n\n"
printf "max_seq_length=$max_seq_length \tmax_epochs=$max_epochs \tbatch_size=80\033[0m"
## args: dir, lr, patience (for early stopping), batch_size, eval_steps, max_train_epochs, max_seq_length, seed
./finetune_all_tasks.sh $model_dir 5e-5 10 80 '0.05' $max_epochs $max_seq_length

# python collect_results.py $model_dir
