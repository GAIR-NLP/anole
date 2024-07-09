import os
import torch
import shutil

from constants_facilitating_image_generation import (
    CHAMELEON_PATH_TORCH,
    ANOLE_PATH_HF,
    ANOLE_PATH_TORCH
)

# Create directories if they do not exist
os.makedirs(ANOLE_PATH_TORCH / 'models/7b', exist_ok=True)
os.makedirs(ANOLE_PATH_TORCH / 'tokenizer', exist_ok=True)

bin_state_dict = torch.load(ANOLE_PATH_HF / "pytorch_model.bin")
print("loaded cheameleon-hf weights.")
pth_state_dict = torch.load(CHAMELEON_PATH_TORCH / "models" / "7b" / "consolidated.pth")
print("loaded cheameleon-torch weights.")

if "lm_head.weight" in bin_state_dict and "output.weight" in pth_state_dict:
    new_weight = bin_state_dict["lm_head.weight"][:65536, :].bfloat16()
    pth_state_dict["output.weight"] = new_weight

torch.save(pth_state_dict, ANOLE_PATH_TORCH / "models/7b/consolidated.pth")

files_to_copy = [
    "models/7b/checklist.chk", 
    "models/7b/config.json", 
    "models/7b/consolidate_params.json", 
    "models/7b/params.json", 
    "tokenizer/checklist.chk", 
    "tokenizer/text_tokenizer.json", 
    "tokenizer/vqgan.ckpt", 
    "tokenizer/vqgan.yaml"
]

for filename in files_to_copy:
    shutil.copy(
        CHAMELEON_PATH_TORCH / filename,
        ANOLE_PATH_TORCH / filename
    )
