## Facilitating Image Generation from Chameleon

To build Anole-7b-v0.1, we conduct fine-tuning on only the image head of Chameleon using around 6,000 image samples from the subset of [laion-art](https://huggingface.co/datasets/fantasyfish/laion-art). To facilitate training convergence, we set a higher-than-usual learning rate (1e-3) and configured the model to output only image logits during training. You can download Anole-7b-v0.1 [here](https://huggingface.co/GAIR/Anole-7b-v0.1). If you want to try facilitating image generation from Chameleon yourself, please follow these steps.

## Steps
1. Download the dataset from [laion-art](https://huggingface.co/datasets/fantasyfish/laion-art) and model from [Chameleon](https://ai.meta.com/resources/models-and-libraries/chameleon-downloads/)

2. Modify [modeling_chameleon.py](https://github.com/GAIR-NLP/PrivateChameleon/blob/main/transformers/src/transformers/models/chameleon/modeling_chameleon.py)
```
# Modify line 1628 and line 1629 of modeling_chameleon.py

# Original Code:
image_tokens = self.model.vocabulary_mapping.image_tokens
logits[:, :, image_tokens] = torch.finfo(logits.dtype).min

# Modified Code:
text_tokens = [i for i in range(0, 4)] + [i for i in range(8198, 65536)]
logits[:, :, text_tokens] = torch.finfo(logits.dtype).min
```

3. Set the constants in `constants_facilitating_image_generation.py`

4. Prepare the finetuning data (you can also directly used the [processed data](https://github.com/GAIR-NLP/PrivateChameleon/blob/main/facilitating_image_generation/dataset_tokenized.jsonl))
```
bash prepare_data.sh
```

5. Convert PyTorch model to Hugging Face model
```
cd ../transformers/src/transformers/models/chameleon/
python convert_chameleon_weights_to_hf.py --model_size 7B --input_dir CHAMELEON_PATH_TORCH --output_dir CHAMELEON_PATH_HF
```

6. train the model using huggingface trainer
```
bash train_image_head.sh
```

7. Inference (Please refer to [Inference](https://github.com/GAIR-NLP/PrivateChameleon?tab=readme-ov-file#inference-on-anole))
