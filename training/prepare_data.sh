#!/bin/bash
PARENT_DIR=$(dirname "$(pwd)")
export PYTHONPATH=$PARENT_DIR

python data_tokenization.py
