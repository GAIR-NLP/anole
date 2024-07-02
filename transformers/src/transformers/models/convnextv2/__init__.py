# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from typing import TYPE_CHECKING

# rely on isort to merge the imports
from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_torch_available,
    is_tf_available,
)


_import_structure = {"configuration_convnextv2": ["ConvNextV2Config"]}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_convnextv2"] = [
        "ConvNextV2ForImageClassification",
        "ConvNextV2Model",
        "ConvNextV2PreTrainedModel",
        "ConvNextV2Backbone",
    ]

try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_convnextv2"] = [
        "TFConvNextV2ForImageClassification",
        "TFConvNextV2Model",
        "TFConvNextV2PreTrainedModel",
    ]

if TYPE_CHECKING:
    from .configuration_convnextv2 import (
        ConvNextV2Config,
    )

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_convnextv2 import (
            ConvNextV2Backbone,
            ConvNextV2ForImageClassification,
            ConvNextV2Model,
            ConvNextV2PreTrainedModel,
        )

    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_convnextv2 import (
            TFConvNextV2ForImageClassification,
            TFConvNextV2Model,
            TFConvNextV2PreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
