import os
import uuid
import torch
import argparse
from PIL import Image
from chameleon.inference.chameleon import ChameleonInferenceModel, Options
from constants import (
    MODEL_7B_PATH,
    TOKENIZER_TEXT_PATH,
    TOKENIZER_IMAGE_CFG_PATH,
    TOKENIZER_IMAGE_PATH,
)
from typing import List

def main(args: argparse.Namespace):
    """Main function to generate images from instructions."""
    
    # Print configuration
    print(f"Instruction: {args.instruction}")
    print(f"Batch size: {args.batch_size}")
    
    # Load Chameleon model
    model = ChameleonInferenceModel(
        MODEL_7B_PATH.as_posix(),
        TOKENIZER_TEXT_PATH.as_posix(),
        TOKENIZER_IMAGE_CFG_PATH.as_posix(),
        TOKENIZER_IMAGE_PATH.as_posix(),
    )
    
    # Generate options
    options = Options()
    options.txt = False
    
    # Prepare batch prompts
    instructions: List[str] = [args.instruction for _ in range(args.batch_size)]
    batch_prompt_ui = []
    for instruction in instructions:
        batch_prompt_ui += [
            [
                {"type": "text", "value": instruction},
                {"type": "sentinel", "value": "<END-OF-TURN>"}
            ],
        ]
    
    # Generate images
    image_tokens: torch.LongTensor = model.generate(
        batch_prompt_ui=batch_prompt_ui,
        options=options
    )
    images: List[Image.Image] =  model.decode_image(image_tokens)

    # Save images
    os.makedirs(args.save_dir, exist_ok=True)
    for instruction, image in zip(instructions, images):
        subdir = instruction.split(' ')[0]
        image_path = os.path.join(args.save_dir, f"{subdir}-{uuid.uuid4()}.png")
        image.save(image_path)
        print(f"Save generated images to {image_path}")

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate images based on text instructions.")
    parser.add_argument("-i", "--instruction", type=str, required=True, help="The instruction for image generation.")
    parser.add_argument("-b", "--batch_size", type=int, default=10, help="The number of images to generate.")
    parser.add_argument("-s", "--save_dir", type=str, default="./outputs/text2image/", help="The directory to save the generated images.")
    args: argparse.Namespace = parser.parse_args()
    return args

if __name__ == "__main__":
    args: argparse.Namespace = parse_arguments()
    main(args)
