## Train Anole on your custom data

You can train Anole on your custom data. Note that the current training code has not been fully verified, but we will continuously update it soon!

## Steps
1. Modify [modeling_chameleon.py](https://github.com/GAIR-NLP/PrivateChameleon/blob/main/transformers/src/transformers/models/chameleon/modeling_chameleon.py)
```
# Modify line 1628 and line 1629 of modeling_chameleon.py

# Original Code:
image_tokens = self.model.vocabulary_mapping.image_tokens
logits[:, :, image_tokens] = torch.finfo(logits.dtype).min

# Modified Code:
# image_tokens = self.model.vocabulary_mapping.image_tokens
# logits[:, :, image_tokens] = torch.finfo(logits.dtype).min
```

2. Prepare your raw finetuning data like [this](https://github.com/GAIR-NLP/PrivateChameleon/blob/main/facilitating_image_generation/dataset_raw.jsonl)

Note: Current code only supports finetuning on one-text segment and one image, we will support multiple interleaved text segments and images finetuning soon.
```
# Example samples
{"text": "Give me an image of Orange juice in a mason glass with an orange cut in half and wooden orange squeezer.", "image": "/path/to/image/1.png"}
{"text": "Give me an image of Chibi_Yukata_Disney_Princesses_by_vulpixfairy-picture", "image": "/path/to/image/2.png"}
```

3. Set the constants in `constants_training.py`

4. Convert raw finetuning data to tokenized data
```
bash prepare_data.sh
```

5. Convert PyTorch model to Hugging Face model
```
cd ../transformers/src/transformers/models/chameleon/
python convert_chameleon_weights_to_hf.py --model_size 7B --input_dir ANOLE_PATH_TORCH --output_dir ANOLE_PATH_HF
```

6. train the model using huggingface trainer
```
bash train.sh
```

7. Convert the huggingface model back to the torch model for inference
```
# specify `ANOLE_PATH_HF_TRAINED` and `ANOLE_PATH_TORCH` in constants_training.py
python bin_to_pth.py
```
