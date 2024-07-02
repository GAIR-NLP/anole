import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ChameleonForCausalLM, Trainer, TrainingArguments
from torch.nn.utils.rnn import pad_sequence
import jsonlines

from constants_facilitating_image_generation import (
    CHAMELEON_PATH_HF,
    ANOLE_PATH_HF,
    DATASET_TOKENIZED_PATH
)

# Define the dataset class
class TokenizedDataset(Dataset):
    def __init__(self, filepath):
        self.tokenized_data = []
        with jsonlines.open(filepath) as reader:
            for obj in reader:
                self.tokenized_data.append(torch.tensor(obj['image_tokens'], dtype=torch.long))
    
    def __len__(self):
        return len(self.tokenized_data)
    
    def __getitem__(self, idx):
        return self.tokenized_data[idx],

# Define custom collate function for DataLoader
def collate_fn(batch):
    batch_inputs = [item[0] for item in batch]
    batch_inputs_padded = pad_sequence(batch_inputs, batch_first=True, padding_value=-100)

    # Create attention masks
    attention_masks = torch.zeros_like(batch_inputs_padded, dtype=torch.long)
    attention_masks = attention_masks.masked_fill(batch_inputs_padded != -100, 1)
   
    return {'input_ids': batch_inputs_padded, 'attention_mask': attention_masks, 'labels': batch_inputs_padded.clone()}

# Initialize the model
model = ChameleonForCausalLM.from_pretrained(CHAMELEON_PATH_HF)
print(model)

# Define the range of weights that should remain trainable
trainable_range = (4, 8196)

# Define a hook to zero out the gradient for weights outside the trainable range during the backward pass
def zero_out_gradient(grad):
    grad[:trainable_range[0], :] = 0
    grad[trainable_range[1] + 1:, :] = 0
    return grad

# Freeze all layers except the output layer
for param in model.parameters():
    param.requires_grad = False
for param in model.lm_head.parameters():
    # Compute the standard deviation for He initialization
    std_dev = (2.0 / 4096) ** 0.5
    # Initialize the specific rows with He initialization
    param[4:8196] = torch.randn((8196 - 4, 4096)) * std_dev
    param.requires_grad = True
    # Register the hook on the weight tensor
    param.register_hook(zero_out_gradient)

# Initialize the dataset
dataset = TokenizedDataset(DATASET_TOKENIZED_PATH.as_posix())

# Define training arguments
training_args = TrainingArguments(
    output_dir=ANOLE_PATH_HF,
    learning_rate=1e-3,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    save_steps=3000,
    fp16=False,
    logging_strategy="steps",
    logging_steps=1,
    deepspeed="ds_config.json"
)

# Initialize the Trainer with custom collate_fn
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collate_fn
)

# Train the model
trainer.train()

# Save the model
torch.save(model.state_dict(), ANOLE_PATH_HF / 'pytorch_model.bin')
