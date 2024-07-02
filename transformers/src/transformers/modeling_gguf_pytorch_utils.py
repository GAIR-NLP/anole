# coding=utf-8
# Copyright 2024 The ggml.ai team and The HuggingFace Inc. team. and pygguf author (github.com/99991)
# https://github.com/99991/pygguf
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from tqdm import tqdm

from .integrations import (
    GGUF_CONFIG_MAPPING,
    GGUF_TENSOR_MAPPING,
    GGUF_TOKENIZER_MAPPING,
    _gguf_parse_value,
    load_dequant_gguf_tensor,
)
from .utils import is_torch_available
from .utils.logging import get_logger


if is_torch_available():
    import torch

logger = get_logger(__name__)


GGUF_TO_TRANSFORMERS_MAPPING = {
    "ignore": {
        "GGUF": {
            "version": "version",
            "tensor_count": "tensor_count",
            "kv_count": "kv_count",
        },
        "general": {"file_type": "file_type", "quantization_version": "quantization_version"},
    },
    "config": GGUF_CONFIG_MAPPING,
    "tensors": GGUF_TENSOR_MAPPING,
    "tokenizer": {"tokenizer": GGUF_TOKENIZER_MAPPING["tokenizer"]},
    "tokenizer_config": {"tokenizer": GGUF_TOKENIZER_MAPPING["tokenizer_config"]},
}

GGUF_SUPPORTED_ARCHITECTURES = list(GGUF_TO_TRANSFORMERS_MAPPING["tensors"].keys())


def read_field(reader, field):
    value = reader.fields[field]
    return [_gguf_parse_value(value.parts[_data_index], value.types) for _data_index in value.data]


def load_gguf_checkpoint(gguf_checkpoint_path, return_tensors=False):
    """
    Load a GGUF file and return a dictionary of parsed parameters containing tensors, the parsed
    tokenizer and config attributes.

    Args:
        gguf_checkpoint_path (`str`):
            The path the to GGUF file to load
        return_tensors (`bool`, defaults to `True`):
            Whether to read the tensors from the file and return them. Not doing so is faster
            and only loads the metadata in memory.
    """
    try:
        from gguf import GGUFReader
    except (ImportError, ModuleNotFoundError):
        logger.error(
            "Loading a GGUF checkpoint in PyTorch, requires both PyTorch and GGUF to be installed. Please see "
            "https://pytorch.org/ and https://github.com/ggerganov/llama.cpp/tree/master/gguf-py for installation instructions."
        )
        raise

    reader = GGUFReader(gguf_checkpoint_path)
    fields = reader.fields
    reader_keys = list(fields.keys())

    parsed_parameters = {k: {} for k in GGUF_TO_TRANSFORMERS_MAPPING}

    architecture = read_field(reader, "general.architecture")[0]
    model_name = read_field(reader, "general.name")

    # in llama.cpp mistral models use the same architecture as llama. We need
    # to add this patch to ensure things work correctly on our side.
    if "llama" in architecture and "mistral" in model_name:
        updated_architecture = "mistral"
    else:
        updated_architecture = architecture

    if architecture not in GGUF_SUPPORTED_ARCHITECTURES:
        raise ValueError(f"Architecture {architecture} not supported")

    # List all key-value pairs in a columnized format
    for gguf_key, field in reader.fields.items():
        gguf_key = gguf_key.replace(architecture, updated_architecture)
        split = gguf_key.split(".")
        prefix = split[0]
        config_key = ".".join(split[1:])

        value = [_gguf_parse_value(field.parts[_data_index], field.types) for _data_index in field.data]

        if len(value) == 1:
            value = value[0]

        if isinstance(value, str) and architecture in value:
            value = value.replace(architecture, updated_architecture)

        for parameter in GGUF_TO_TRANSFORMERS_MAPPING:
            parameter_renames = GGUF_TO_TRANSFORMERS_MAPPING[parameter]
            if prefix in parameter_renames and config_key in parameter_renames[prefix]:
                renamed_config_key = parameter_renames[prefix][config_key]
                if renamed_config_key == -1:
                    continue

                if renamed_config_key is not None:
                    parsed_parameters[parameter][renamed_config_key] = value

                if gguf_key in reader_keys:
                    reader_keys.remove(gguf_key)

        if gguf_key in reader_keys:
            logger.info(f"Some keys were not parsed and added into account {gguf_key} | {value}")

    if return_tensors:
        tensor_key_mapping = GGUF_TO_TRANSFORMERS_MAPPING["tensors"][architecture]

        for tensor in tqdm(reader.tensors, desc="Converting and de-quantizing GGUF tensors..."):
            renamed_tensor_name = tensor.name

            for tensor_name_mapping in GGUF_TO_TRANSFORMERS_MAPPING["tensors"]:
                if tensor_name_mapping in renamed_tensor_name:
                    renamed_tensor_name = renamed_tensor_name.replace(
                        tensor_name_mapping, GGUF_TO_TRANSFORMERS_MAPPING["tensors"][tensor_name_mapping]
                    )

            shape = tensor.shape
            name = tensor.name

            weights = load_dequant_gguf_tensor(shape=shape, ggml_type=tensor.tensor_type, data=tensor.data)

            if architecture == "llama" and (".attn_k." in name or ".attn_q." in name):
                num_heads = parsed_parameters["config"]["num_attention_heads"]
                tmp_shape = (int(shape[-1] // num_heads // 2), num_heads, 2, shape[0])
                weights = weights.reshape(tmp_shape)
                weights = weights.transpose(0, 2, 1, 3)
                weights = weights.reshape(shape[::-1])

            for tensor_name in tensor_key_mapping:
                if tensor_name in name:
                    name = name.replace(tensor_name, tensor_key_mapping[tensor_name])

            # Use copy to avoid errors with numpy and pytorch
            parsed_parameters["tensors"][name] = torch.from_numpy(np.copy(weights))

    if len(reader_keys) > 0:
        logger.info(f"Some keys of the GGUF file were not considered: {reader_keys}")

    return parsed_parameters
