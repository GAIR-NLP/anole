import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

# Raw dataset (specify the path of your raw dataset)
DATASET_RAW_PATH = Path("./dataset_raw.jsonl")

# Tokenized dataset (specify the path that you want to store your tokenized dataset)
DATASET_TOKENIZED_PATH = Path("./dataset_tokenized.jsonl")

# Tokenized dataset (specify the path that you want to store your images)
DATASET_IMAGE_PATH = Path("./images/")

# Anole torch path (specify the path that you want to store your Anole torch checkpoint)
ANOLE_PATH_TORCH = Path("./model/anole/")

# Anole HF path (specify the path that you want to store your Anole hugging face checkpoint)
ANOLE_PATH_HF = Path("./model/anole-hf/")

# Anole HF path (specify the path that you want to store your fine-tuned Anole hugging face checkpoint)
ANOLE_PATH_HF_TRAINED = Path("./model/anole-hf_trained/")
