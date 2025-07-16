import os
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
from typing import List, Tuple

def split_token_sequence(
    tokens: torch.LongTensor,
    boi: int,
    eoi: int
) -> List[Tuple[str, torch.LongTensor]]:
    """
    Split a sequence of tokens into text and image segments.
    
    Args:
        tokens (torch.LongTensor): The token sequence.
        boi (int): Begin of image token.
        eoi (int): End of image token.
    
    Returns:
        List[Tuple[str, torch.LongTensor]]: List of tuples indicating segment type and tokens.
    """
    batch_size, _ = tokens.shape
    assert batch_size == 1, "Batch size must be 1"
    
    device = tokens.device
    tokens = tokens[0]  # remove batch dimension
    tokens = tokens.to(device)
    segments = []
    current_segment = []
    in_image_seg = False

    for token in tokens:
        if token == boi:
            # if entering an image segment, save the current text segment (if any)
            if current_segment:
                segments.append(("text_seg", torch.tensor(current_segment, dtype=tokens.dtype, device=device).reshape(1, -1)))
                current_segment = []
            in_image_seg = True
        elif token == eoi and in_image_seg:
            # if exiting an image segment, save the current image segment
            segments.append(("image_seg", torch.tensor(current_segment, dtype=tokens.dtype, device=device).reshape(1, -1)))
            current_segment = []
            in_image_seg = False
        else:
            current_segment.append(token)
    # save any remaining tokens
    if current_segment:
        if in_image_seg:
            segments.append(("image_seg", torch.tensor(current_segment, dtype=tokens.dtype, device=device).reshape(1, -1)))
        else:
            segments.append(("text_seg", torch.tensor(current_segment, dtype=tokens.dtype, device=device).reshape(1, -1)))
    return segments

def main(args: argparse.Namespace):
    """Main function to generate and process model output."""
    # Load Chameleon model
    unquantized_model = ChameleonInferenceModel(
        MODEL_7B_PATH.as_posix(),
        TOKENIZER_TEXT_PATH.as_posix(),
        TOKENIZER_IMAGE_CFG_PATH.as_posix(),
        TOKENIZER_IMAGE_PATH.as_posix(),
    )
    model = torch.quantization.quantize_dynamic(
        unquantized_model,  # The model to be quantized
        {torch.nn.Linear, torch.nn.LSTM},  # Layers to be dynamically quantized
        dtype=torch.qint8  # Data type for quantization
    )
    # Print model configuration
    print(f"Model path: {MODEL_7B_PATH}")
    print(f"Text tokenizer path: {TOKENIZER_TEXT_PATH}")
    print(f"Image tokenizer config path: {TOKENIZER_IMAGE_CFG_PATH}")
    print(f"Image tokenizer path: {TOKENIZER_IMAGE_PATH}")
    # Generate options
    options = Options()
    # Prepare prompt
    instructions = [args.instruction]
    batch_prompt_ui = []
    for instruction in instructions:
        if isinstance(instruction, Tuple):
            inst, image_path = instruction
            batch_prompt_ui += [
                [
                    {"type": "image", "value": f"file:{image_path}"},
                    {"type": "text", "value": inst}
                ],
            ]
        else:
            batch_prompt_ui += [
                [
                    {"type": "text", "value": instruction}
                ],
            ]
    # generate
    tokens: torch.LongTensor = model.generate(
        batch_prompt_ui=batch_prompt_ui,
        options=options
    )
    # split
    boi, eoi = model.vocab.begin_image, model.vocab.end_image   # 8197(boi), 8196(eoi)
    segments = split_token_sequence(tokens, boi, eoi)
    # decode
    os.makedirs(args.save_dir, exist_ok=True)
    for seg_id, (seg_type, seg_tokens) in enumerate(segments):
        if seg_type == "image_seg":
            assert seg_tokens.shape[1] == 1024
            img: Image = model.decode_image(seg_tokens)[0]
            image_path = os.path.join(args.save_dir, f"{seg_id}.png")
            img.save(image_path)
            print(f"<img: {image_path}>")
        else:
            assert seg_type == "text_seg"
            decoded_text = model.decode_text(seg_tokens)[0]
            print(decoded_text)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate interleaved image-text content based on text instructions.")
    parser.add_argument("-i", "--instruction", type=str, required=True, help="The instruction for interleaved image-text generation.")
    parser.add_argument("-s", "--save_dir", type=str, default="./outputs/interleaved/", help="The directory to save the generated images.")
    args: argparse.Namespace = parser.parse_args()
    return args

if __name__ == "__main__":
    args: argparse.Namespace = parse_arguments()
    main(args)
