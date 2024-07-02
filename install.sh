#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

cd transformers
pip install -e .
cd ..
pip install -r requirements.txt

# Note: You can only run this part if you only want to do inference.
cd chameleon
pip install -e .
