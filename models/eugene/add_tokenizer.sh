#!/bin/bash

# Paths to the source directories
GPT_SOURCE="../gpt-10k-tok"
ROBERTA_SOURCE="../roberta-10k-tok"

# Find all subdirectories within the 'models' directory
for dir in */; do
  # Check if the directory name contains 'gpt'
  if [[ $dir == *gpt* ]]; then
    cp -r "$GPT_SOURCE"/* "$dir"
  # Check if the directory name contains 'roberta'
  elif [[ $dir == *roberta* ]]; then
    cp -r "$ROBERTA_SOURCE"/* "$dir"
  fi
done