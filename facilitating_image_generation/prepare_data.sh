#!/bin/bash
PARENT_DIR=$(dirname "$(pwd)")
export PYTHONPATH=$PARENT_DIR

python read_parquet.py
python data_tokenization.py
