#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

pip install -r requirements.txt
cd transformers
pip install -e .