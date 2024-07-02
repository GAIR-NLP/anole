import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

# Specify the list of paths of the parquet files (that stores the dataset)
PARQUET_PATHS = [
    "/path/to/your/laion-art/data/train-00001-of-00024-63870cfe2fd3db14.parquet",
    "/path/to/your/laion-art/data/train-00002-of-00024-eeff6012f4c4b16e.parquet",
    "/path/to/your/laion-art/data/train-00003-of-00024-2eeae88ce2f4a622.parquet",
    "/path/to/your/laion-art/data/train-00004-of-00024-ebf603b95d754eef.parquet",
    "/path/to/your/laion-art/data/train-00005-of-00024-d575b0800f88101b.parquet",
    "/path/to/your/laion-art/data/train-00006-of-00024-a004074791510f33.parquet",
    "/path/to/your/laion-art/data/train-00007-of-00024-e7301ae04f696809.parquet",
]

# Raw dataset (specify the path that you want to store your raw dataset)
DATASET_RAW_PATH = Path("./dataset_raw.jsonl")

# Tokenized dataset (specify the path that you want to store your tokenized dataset)
DATASET_TOKENIZED_PATH = Path("./dataset_tokenized.jsonl")

# Tokenized dataset (specify the path that you want to store your images extracted from parquet files)
DATASET_IMAGE_PATH = Path("./images/")

# Chameleon path (Chameleon checkpoint path)
CHAMELEON_PATH_TORCH = Path("/path/to/your/meta-chameleon/")

# Chameleon HF path (specify the path that you want to store your chameleon hugging face checkpoint)
CHAMELEON_PATH_HF = Path("./model/meta-chameleon-hf/")

# Anole torch path (specify the path that you want to store your Anole torch checkpoint)
ANOLE_PATH_TORCH = Path("./model/anole/")

# Anole HF path (specify the path that you want to store your Anole hugging face checkpoint)
ANOLE_PATH_HF = Path("./model/anole-hf/")
