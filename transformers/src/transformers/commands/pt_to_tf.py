# Copyright 2022 The HuggingFace Team. All rights reserved.
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

import inspect
import os
from argparse import ArgumentParser, Namespace
from importlib import import_module

import huggingface_hub
import numpy as np
from packaging import version

from .. import (
    FEATURE_EXTRACTOR_MAPPING,
    IMAGE_PROCESSOR_MAPPING,
    PROCESSOR_MAPPING,
    TOKENIZER_MAPPING,
    AutoConfig,
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoProcessor,
    AutoTokenizer,
    is_datasets_available,
    is_tf_available,
    is_torch_available,
)
from ..utils import TF2_WEIGHTS_INDEX_NAME, TF2_WEIGHTS_NAME, logging
from . import BaseTransformersCLICommand


if is_tf_available():
    import tensorflow as tf

    tf.config.experimental.enable_tensor_float_32_execution(False)

if is_torch_available():
    import torch

if is_datasets_available():
    from datasets import load_dataset


MAX_ERROR = 5e-5  # larger error tolerance than in our internal tests, to avoid flaky user-facing errors


def convert_command_factory(args: Namespace):
    """
    Factory function used to convert a model PyTorch checkpoint in a TensorFlow 2 checkpoint.

    Returns: ServeCommand
    """
    return PTtoTFCommand(
        args.model_name,
        args.local_dir,
        args.max_error,
        args.new_weights,
        args.no_pr,
        args.push,
        args.extra_commit_description,
        args.override_model_class,
    )


class PTtoTFCommand(BaseTransformersCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        """
        Register this command to argparse so it's available for the transformer-cli

        Args:
            parser: Root parser to register command-specific arguments
        """
        train_parser = parser.add_parser(
            "pt-to-tf",
            help=(
                "CLI tool to run convert a transformers model from a PyTorch checkpoint to a TensorFlow checkpoint."
                " Can also be used to validate existing weights without opening PRs, with --no-pr."
            ),
        )
        train_parser.add_argument(
            "--model-name",
            type=str,
            required=True,
            help="The model name, including owner/organization, as seen on the hub.",
        )
        train_parser.add_argument(
            "--local-dir",
            type=str,
            default="",
            help="Optional local directory of the model repository. Defaults to /tmp/{model_name}",
        )
        train_parser.add_argument(
            "--max-error",
            type=float,
            default=MAX_ERROR,
            help=(
                f"Maximum error tolerance. Defaults to {MAX_ERROR}. This flag should be avoided, use at your own risk."
            ),
        )
        train_parser.add_argument(
            "--new-weights",
            action="store_true",
            help="Optional flag to create new TensorFlow weights, even if they already exist.",
        )
        train_parser.add_argument(
            "--no-pr", action="store_true", help="Optional flag to NOT open a PR with converted weights."
        )
        train_parser.add_argument(
            "--push",
            action="store_true",
            help="Optional flag to push the weights directly to `main` (requires permissions)",
        )
        train_parser.add_argument(
            "--extra-commit-description",
            type=str,
            default="",
            help="Optional additional commit description to use when opening a PR (e.g. to tag the owner).",
        )
        train_parser.add_argument(
            "--override-model-class",
            type=str,
            default=None,
            help="If you think you know better than the auto-detector, you can specify the model class here. "
            "Can be either an AutoModel class or a specific model class like BertForSequenceClassification.",
        )
        train_parser.set_defaults(func=convert_command_factory)

    @staticmethod
    def find_pt_tf_differences(pt_outputs, tf_outputs):
        """
        Compares the TensorFlow and PyTorch outputs, returning a dictionary with all tensor differences.
        """
        # 1. All output attributes must be the same
        pt_out_attrs = set(pt_outputs.keys())
        tf_out_attrs = set(tf_outputs.keys())
        if pt_out_attrs != tf_out_attrs:
            raise ValueError(
                f"The model outputs have different attributes, aborting. (Pytorch: {pt_out_attrs}, TensorFlow:"
                f" {tf_out_attrs})"
            )

        # 2. For each output attribute, computes the difference
        def _find_pt_tf_differences(pt_out, tf_out, differences, attr_name=""):
            # If the current attribute is a tensor, it is a leaf and we make the comparison. Otherwise, we will dig in
            # recursivelly, keeping the name of the attribute.
            if isinstance(pt_out, torch.Tensor):
                tensor_difference = np.max(np.abs(pt_out.numpy() - tf_out.numpy()))
                differences[attr_name] = tensor_difference
            else:
                root_name = attr_name
                for i, pt_item in enumerate(pt_out):
                    # If it is a named attribute, we keep the name. Otherwise, just its index.
                    if isinstance(pt_item, str):
                        branch_name = root_name + pt_item
                        tf_item = tf_out[pt_item]
                        pt_item = pt_out[pt_item]
                    else:
                        branch_name = root_name + f"[{i}]"
                        tf_item = tf_out[i]
                    differences = _find_pt_tf_differences(pt_item, tf_item, differences, branch_name)

            return differences

        return _find_pt_tf_differences(pt_outputs, tf_outputs, {})

    def __init__(
        self,
        model_name: str,
        local_dir: str,
        max_error: float,
        new_weights: bool,
        no_pr: bool,
        push: bool,
        extra_commit_description: str,
        override_model_class: str,
        *args,
    ):
        self._logger = logging.get_logger("transformers-cli/pt_to_tf")
        self._model_name = model_name
        self._local_dir = local_dir if local_dir else os.path.join("/tmp", model_name)
        self._max_error = max_error
        self._new_weights = new_weights
        self._no_pr = no_pr
        self._push = push
        self._extra_commit_description = extra_commit_description
        self._override_model_class = override_model_class

    def get_inputs(self, pt_model, tf_dummy_inputs, config):
        """
        Returns the right inputs for the model, based on its signature.
        """

        def _get_audio_input():
            ds = load_dataset(
                "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation", trust_remote_code=True
            )
            speech_samples = ds.sort("id").select(range(2))[:2]["audio"]
            raw_samples = [x["array"] for x in speech_samples]
            return raw_samples

        model_config_class = type(pt_model.config)
        if model_config_class in PROCESSOR_MAPPING:
            processor = AutoProcessor.from_pretrained(self._local_dir)
            if model_config_class in TOKENIZER_MAPPING and processor.tokenizer.pad_token is None:
                processor.tokenizer.pad_token = processor.tokenizer.eos_token
        elif model_config_class in IMAGE_PROCESSOR_MAPPING:
            processor = AutoImageProcessor.from_pretrained(self._local_dir)
        elif model_config_class in FEATURE_EXTRACTOR_MAPPING:
            processor = AutoFeatureExtractor.from_pretrained(self._local_dir)
        elif model_config_class in TOKENIZER_MAPPING:
            processor = AutoTokenizer.from_pretrained(self._local_dir)
            if processor.pad_token is None:
                processor.pad_token = processor.eos_token
        else:
            raise ValueError(f"Unknown data processing type (model config type: {model_config_class})")

        model_forward_signature = set(inspect.signature(pt_model.forward).parameters.keys())
        processor_inputs = {}
        if "input_ids" in model_forward_signature:
            processor_inputs.update(
                {
                    "text": ["Hi there!", "I am a batch with more than one row and different input lengths."],
                    "padding": True,
                    "truncation": True,
                }
            )
        if "pixel_values" in model_forward_signature:
            sample_images = load_dataset("uoft-cs/cifar10", "plain_text", split="test")[:2]["img"]  # no-script
            processor_inputs.update({"images": sample_images})
        if "input_features" in model_forward_signature:
            feature_extractor_signature = inspect.signature(processor.feature_extractor).parameters
            # Pad to the largest input length by default but take feature extractor default
            # padding value if it exists e.g. "max_length" and is not False or None
            if "padding" in feature_extractor_signature:
                default_strategy = feature_extractor_signature["padding"].default
                if default_strategy is not False and default_strategy is not None:
                    padding_strategy = default_strategy
                else:
                    padding_strategy = True
            else:
                padding_strategy = True
            processor_inputs.update({"audio": _get_audio_input(), "padding": padding_strategy})
        if "input_values" in model_forward_signature:  # Wav2Vec2 audio input
            processor_inputs.update({"audio": _get_audio_input(), "padding": True})
        pt_input = processor(**processor_inputs, return_tensors="pt")
        tf_input = processor(**processor_inputs, return_tensors="tf")

        # Extra input requirements, in addition to the input modality
        if (
            config.is_encoder_decoder
            or (hasattr(pt_model, "encoder") and hasattr(pt_model, "decoder"))
            or "decoder_input_ids" in tf_dummy_inputs
        ):
            decoder_input_ids = np.asarray([[1], [1]], dtype=int) * (pt_model.config.decoder_start_token_id or 0)
            pt_input.update({"decoder_input_ids": torch.tensor(decoder_input_ids)})
            tf_input.update({"decoder_input_ids": tf.convert_to_tensor(decoder_input_ids)})

        return pt_input, tf_input

    def run(self):
        self._logger.warning(
            "\n\nConverting PyTorch weights to TensorFlow is deprecated and will be removed in v4.43. "
            "Instead, we recommend that you convert PyTorch weights to Safetensors, an improved "
            "format that can be loaded by any framework, including TensorFlow. For more information, "
            "please see the Safetensors conversion guide: "
            "https://huggingface.co/docs/safetensors/en/convert-weights\n\n"
        )
        # hub version 0.9.0 introduced the possibility of programmatically opening PRs with normal write tokens.
        if version.parse(huggingface_hub.__version__) < version.parse("0.9.0"):
            raise ImportError(
                "The huggingface_hub version must be >= 0.9.0 to use this command. Please update your huggingface_hub"
                " installation."
            )
        else:
            from huggingface_hub import Repository, create_commit
            from huggingface_hub._commit_api import CommitOperationAdd

        # Fetch remote data
        repo = Repository(local_dir=self._local_dir, clone_from=self._model_name)

        # Load config and get the appropriate architecture -- the latter is needed to convert the head's weights
        config = AutoConfig.from_pretrained(self._local_dir)
        architectures = config.architectures
        if self._override_model_class is not None:
            if self._override_model_class.startswith("TF"):
                architectures = [self._override_model_class[2:]]
            else:
                architectures = [self._override_model_class]
            try:
                pt_class = getattr(import_module("transformers"), architectures[0])
            except AttributeError:
                raise ValueError(f"Model class {self._override_model_class} not found in transformers.")
            try:
                tf_class = getattr(import_module("transformers"), "TF" + architectures[0])
            except AttributeError:
                raise ValueError(f"TF model class TF{self._override_model_class} not found in transformers.")
        elif architectures is None:  # No architecture defined -- use auto classes
            pt_class = getattr(import_module("transformers"), "AutoModel")
            tf_class = getattr(import_module("transformers"), "TFAutoModel")
            self._logger.warning("No detected architecture, using AutoModel/TFAutoModel")
        else:  # Architecture defined -- use it
            if len(architectures) > 1:
                raise ValueError(f"More than one architecture was found, aborting. (architectures = {architectures})")
            self._logger.warning(f"Detected architecture: {architectures[0]}")
            pt_class = getattr(import_module("transformers"), architectures[0])
            try:
                tf_class = getattr(import_module("transformers"), "TF" + architectures[0])
            except AttributeError:
                raise AttributeError(f"The TensorFlow equivalent of {architectures[0]} doesn't exist in transformers.")

        # Check the TF dummy inputs to see what keys we need in the forward pass
        tf_from_pt_model = tf_class.from_config(config)
        tf_dummy_inputs = tf_from_pt_model.dummy_inputs

        del tf_from_pt_model  # Try to keep only one model in memory at a time

        # Load the model and get some basic inputs
        pt_model = pt_class.from_pretrained(self._local_dir)
        pt_model.eval()

        pt_input, tf_input = self.get_inputs(pt_model, tf_dummy_inputs, config)

        with torch.no_grad():
            pt_outputs = pt_model(**pt_input, output_hidden_states=True)
        del pt_model  # will no longer be used, and may have a large memory footprint

        tf_from_pt_model = tf_class.from_pretrained(self._local_dir, from_pt=True)
        tf_from_pt_outputs = tf_from_pt_model(**tf_input, output_hidden_states=True, training=False)

        # Confirms that cross loading PT weights into TF worked.
        crossload_differences = self.find_pt_tf_differences(pt_outputs, tf_from_pt_outputs)
        output_differences = {k: v for k, v in crossload_differences.items() if "hidden" not in k}
        hidden_differences = {k: v for k, v in crossload_differences.items() if "hidden" in k}
        if len(output_differences) == 0 and architectures is not None:
            raise ValueError(
                f"Something went wrong -- the config file has architectures ({architectures}), but no model head"
                " output was found. All outputs start with 'hidden'"
            )
        max_crossload_output_diff = max(output_differences.values()) if output_differences else 0.0
        max_crossload_hidden_diff = max(hidden_differences.values())
        if max_crossload_output_diff > self._max_error or max_crossload_hidden_diff > self._max_error:
            raise ValueError(
                "The cross-loaded TensorFlow model has different outputs, something went wrong!\n"
                + f"\nList of maximum output differences above the threshold ({self._max_error}):\n"
                + "\n".join([f"{k}: {v:.3e}" for k, v in output_differences.items() if v > self._max_error])
                + f"\n\nList of maximum hidden layer differences above the threshold ({self._max_error}):\n"
                + "\n".join([f"{k}: {v:.3e}" for k, v in hidden_differences.items() if v > self._max_error])
            )

        # Save the weights in a TF format (if needed) and confirms that the results are still good
        tf_weights_path = os.path.join(self._local_dir, TF2_WEIGHTS_NAME)
        tf_weights_index_path = os.path.join(self._local_dir, TF2_WEIGHTS_INDEX_NAME)
        if (not os.path.exists(tf_weights_path) and not os.path.exists(tf_weights_index_path)) or self._new_weights:
            tf_from_pt_model.save_pretrained(self._local_dir)
        del tf_from_pt_model  # will no longer be used, and may have a large memory footprint

        tf_model = tf_class.from_pretrained(self._local_dir)
        tf_outputs = tf_model(**tf_input, output_hidden_states=True)

        conversion_differences = self.find_pt_tf_differences(pt_outputs, tf_outputs)
        output_differences = {k: v for k, v in conversion_differences.items() if "hidden" not in k}
        hidden_differences = {k: v for k, v in conversion_differences.items() if "hidden" in k}
        if len(output_differences) == 0 and architectures is not None:
            raise ValueError(
                f"Something went wrong -- the config file has architectures ({architectures}), but no model head"
                " output was found. All outputs start with 'hidden'"
            )
        max_conversion_output_diff = max(output_differences.values()) if output_differences else 0.0
        max_conversion_hidden_diff = max(hidden_differences.values())
        if max_conversion_output_diff > self._max_error or max_conversion_hidden_diff > self._max_error:
            raise ValueError(
                "The converted TensorFlow model has different outputs, something went wrong!\n"
                + f"\nList of maximum output differences above the threshold ({self._max_error}):\n"
                + "\n".join([f"{k}: {v:.3e}" for k, v in output_differences.items() if v > self._max_error])
                + f"\n\nList of maximum hidden layer differences above the threshold ({self._max_error}):\n"
                + "\n".join([f"{k}: {v:.3e}" for k, v in hidden_differences.items() if v > self._max_error])
            )

        commit_message = "Update TF weights" if self._new_weights else "Add TF weights"
        if self._push:
            repo.git_add(auto_lfs_track=True)
            repo.git_commit(commit_message)
            repo.git_push(blocking=True)  # this prints a progress bar with the upload
            self._logger.warning(f"TF weights pushed into {self._model_name}")
        elif not self._no_pr:
            self._logger.warning("Uploading the weights into a new PR...")
            commit_descrition = (
                "Model converted by the [`transformers`' `pt_to_tf`"
                " CLI](https://github.com/huggingface/transformers/blob/main/src/transformers/commands/pt_to_tf.py). "
                "All converted model outputs and hidden layers were validated against its PyTorch counterpart.\n\n"
                f"Maximum crossload output difference={max_crossload_output_diff:.3e}; "
                f"Maximum crossload hidden layer difference={max_crossload_hidden_diff:.3e};\n"
                f"Maximum conversion output difference={max_conversion_output_diff:.3e}; "
                f"Maximum conversion hidden layer difference={max_conversion_hidden_diff:.3e};\n"
            )
            if self._max_error > MAX_ERROR:
                commit_descrition += (
                    f"\n\nCAUTION: The maximum admissible error was manually increased to {self._max_error}!"
                )
            if self._extra_commit_description:
                commit_descrition += "\n\n" + self._extra_commit_description

            # sharded model -> adds all related files (index and .h5 shards)
            if os.path.exists(tf_weights_index_path):
                operations = [
                    CommitOperationAdd(path_in_repo=TF2_WEIGHTS_INDEX_NAME, path_or_fileobj=tf_weights_index_path)
                ]
                for shard_path in tf.io.gfile.glob(self._local_dir + "/tf_model-*.h5"):
                    operations += [
                        CommitOperationAdd(path_in_repo=os.path.basename(shard_path), path_or_fileobj=shard_path)
                    ]
            else:
                operations = [CommitOperationAdd(path_in_repo=TF2_WEIGHTS_NAME, path_or_fileobj=tf_weights_path)]

            hub_pr_url = create_commit(
                repo_id=self._model_name,
                operations=operations,
                commit_message=commit_message,
                commit_description=commit_descrition,
                repo_type="model",
                create_pr=True,
            ).pr_url
            self._logger.warning(f"PR open in {hub_pr_url}")
