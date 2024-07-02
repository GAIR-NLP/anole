import io
import os
import json
import random
import pandas as pd
from tqdm import tqdm
from PIL import Image

from facilitating_image_generation.constants_facilitating_image_generation import (
    PARQUET_PATHS,
    DATASET_IMAGE_PATH,
    DATASET_RAW_PATH
)

file_paths = PARQUET_PATHS

# Load all Parquet files into a single DataFrame
dfs = [pd.read_parquet(fp) for fp in file_paths]
df = pd.concat(dfs, ignore_index=True)

# List of prefixes
prefixes = ["Give me an image of", "Generate an image of", "Draw a picture of"]

# Function to add padding to base64 strings
def add_padding(base64_str):
    return base64_str + '=' * (-len(base64_str) % 4)

# Define the output directory
output_dir = DATASET_IMAGE_PATH
os.makedirs(output_dir, exist_ok=True)

# Initialize a list to hold the metadata
metadata = []

# Iterate over the dataframe rows
for index, row in tqdm(df.iterrows(), desc="Extracte images from parquet files"):
    # pdb.set_trace()
    image_bytes = row["image"]["bytes"]
    image_stream = io.BytesIO(image_bytes)
    image = Image.open(image_stream)
    
    # Define the output path for the image
    image_path = os.path.join(output_dir, f"{index}.png")
    image.save(image_path)
    
    # Randomly choose a prefix and add it to the text
    text_with_prefix = f"{random.choice(prefixes)} {row['text']}"
    
    # Create the metadata entry
    metadata_entry = {
        "text": text_with_prefix,
        "image": image_path,
        "aesthetic": row["aesthetic"]
    }
    
    # Add the metadata entry to the list
    metadata.append(metadata_entry)

# Define the output path for the JSONL file
jsonl_path = DATASET_RAW_PATH

# Write the metadata to a JSONL file
with open(jsonl_path, "w") as jsonl_file:
    for entry in metadata:
        jsonl_file.write(json.dumps(entry) + "\n")

print("Images and metadata have been saved successfully.")
