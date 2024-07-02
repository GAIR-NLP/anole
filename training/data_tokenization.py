import os
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from PIL import Image
from chameleon.inference.chameleon import ChameleonInferenceModel

from training.constants_training import (
    ANOLE_PATH_TORCH,
    DATASET_RAW_PATH,
    DATASET_TOKENIZED_PATH
)

if __name__ == "__main__":
    model = ChameleonInferenceModel(
            (ANOLE_PATH_TORCH / "models" / "7b").as_posix(),
            (ANOLE_PATH_TORCH / "tokenizer" / "text_tokenizer.json").as_posix(),
            (ANOLE_PATH_TORCH / "tokenizer" / "vqgan.yaml").as_posix(),
            (ANOLE_PATH_TORCH / "tokenizer" / "vqgan.ckpt").as_posix(),
        )
    
    output_data = model.sft_tokenization(DATASET_RAW_PATH)
    output_file_path = DATASET_TOKENIZED_PATH

    with open(output_file_path, 'w') as output_file:
        for entry in output_data:
            output_file.write(json.dumps(entry) + '\n')
