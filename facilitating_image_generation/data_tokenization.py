import os
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from chameleon.inference.chameleon import ChameleonInferenceModel
from facilitating_image_generation.constants_facilitating_image_generation import (
    CHAMELEON_PATH_TORCH,
    DATASET_RAW_PATH,
    DATASET_TOKENIZED_PATH
)

if __name__ == "__main__":
    model = ChameleonInferenceModel(
            (CHAMELEON_PATH_TORCH / "models" / "7b").as_posix(),
            (CHAMELEON_PATH_TORCH / "tokenizer" / "text_tokenizer.json").as_posix(),
            (CHAMELEON_PATH_TORCH / "tokenizer" / "vqgan.yaml").as_posix(),
            (CHAMELEON_PATH_TORCH / "tokenizer" / "vqgan.ckpt").as_posix(),
        )
    print("load model successfully.")
    
    output_data = model.sft_tokenization(DATASET_RAW_PATH)
    output_file_path = DATASET_TOKENIZED_PATH

    with open(output_file_path, 'w') as output_file:
        for entry in output_data:
            output_file.write(json.dumps(entry) + '\n')
