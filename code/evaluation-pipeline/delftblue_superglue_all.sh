#!/bin/bash

# Check if the directory path is provided as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 path_to_directory"
  exit 1
fi

# Get the directory path from the argument
PARENT_DIR="$1"

# Check if the provided path is a directory
if [ ! -d "$PARENT_DIR" ]; then
  echo "Error: $PARENT_DIR is not a directory"
  exit 1
fi

# Iterate over each subdirectory in the given directory
for SUBDIR in "$PARENT_DIR"/*/; do
  if [ -d "$SUBDIR" ]; then
    sbatch delftblue_superglue.sh "$SUBDIR"
  fi
done