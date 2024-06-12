#!/bin/bash

# Path to the MODEL directory containing model subdirectories
MODEL_DIR=$1

# Find all subdirectories within the specified MODEL directory
for dir in "$MODEL_DIR"/*/; do
  # Check if the directory name contains 'gpt'
  if [[ $dir == *gpt* ]]; then
    python babylm_eval.py "$dir" decoder
  # Check if the directory name contains 'roberta'
  elif [[ $dir == *roberta* ]]; then
    python babylm_eval.py "$dir" encoder
  fi
done