# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import dataclass, field
from typing import Tuple

from ..utils import (
    cached_property,
    is_torch_available,
    is_torch_xla_available,
    is_torch_xpu_available,
    logging,
    requires_backends,
)
from .benchmark_args_utils import BenchmarkArguments


if is_torch_available():
    import torch

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm


logger = logging.get_logger(__name__)


@dataclass
class PyTorchBenchmarkArguments(BenchmarkArguments):
    deprecated_args = [
        "no_inference",
        "no_cuda",
        "no_tpu",
        "no_speed",
        "no_memory",
        "no_env_print",
        "no_multi_process",
    ]

    def __init__(self, **kwargs):
        """
        This __init__ is there for legacy code. When removing deprecated args completely, the class can simply be
        deleted
        """
        for deprecated_arg in self.deprecated_args:
            if deprecated_arg in kwargs:
                positive_arg = deprecated_arg[3:]
                setattr(self, positive_arg, not kwargs.pop(deprecated_arg))
                logger.warning(
                    f"{deprecated_arg} is depreciated. Please use --no_{positive_arg} or"
                    f" {positive_arg}={kwargs[positive_arg]}"
                )

        self.torchscript = kwargs.pop("torchscript", self.torchscript)
        self.torch_xla_tpu_print_metrics = kwargs.pop("torch_xla_tpu_print_metrics", self.torch_xla_tpu_print_metrics)
        self.fp16_opt_level = kwargs.pop("fp16_opt_level", self.fp16_opt_level)
        super().__init__(**kwargs)

    torchscript: bool = field(default=False, metadata={"help": "Trace the models using torchscript"})
    torch_xla_tpu_print_metrics: bool = field(default=False, metadata={"help": "Print Xla/PyTorch tpu metrics"})
    fp16_opt_level: str = field(
        default="O1",
        metadata={
            "help": (
                "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. "
                "See details at https://nvidia.github.io/apex/amp.html"
            )
        },
    )

    @cached_property
    def _setup_devices(self) -> Tuple["torch.device", int]:
        requires_backends(self, ["torch"])
        logger.info("PyTorch: setting up devices")
        if not self.cuda:
            device = torch.device("cpu")
            n_gpu = 0
        elif is_torch_xla_available():
            device = xm.xla_device()
            n_gpu = 0
        elif is_torch_xpu_available():
            device = torch.device("xpu")
            n_gpu = torch.xpu.device_count()
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            n_gpu = torch.cuda.device_count()
        return device, n_gpu

    @property
    def is_tpu(self):
        return is_torch_xla_available() and self.tpu

    @property
    def device_idx(self) -> int:
        requires_backends(self, ["torch"])
        # TODO(PVP): currently only single GPU is supported
        return torch.cuda.current_device()

    @property
    def device(self) -> "torch.device":
        requires_backends(self, ["torch"])
        return self._setup_devices[0]

    @property
    def n_gpu(self):
        requires_backends(self, ["torch"])
        return self._setup_devices[1]

    @property
    def is_gpu(self):
        return self.n_gpu > 0
