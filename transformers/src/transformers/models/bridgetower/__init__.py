# Copyright 2023 The Intel Labs Team Authors, The Microsoft Research Team Authors and HuggingFace Inc. team. All rights reserved.
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

from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available


_import_structure = {
    "configuration_bridgetower": [
        "BridgeTowerConfig",
        "BridgeTowerTextConfig",
        "BridgeTowerVisionConfig",
    ],
    "processing_bridgetower": ["BridgeTowerProcessor"],
}

try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["image_processing_bridgetower"] = ["BridgeTowerImageProcessor"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_bridgetower"] = [
        "BridgeTowerForContrastiveLearning",
        "BridgeTowerForImageAndTextRetrieval",
        "BridgeTowerForMaskedLM",
        "BridgeTowerModel",
        "BridgeTowerPreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_bridgetower import (
        BridgeTowerConfig,
        BridgeTowerTextConfig,
        BridgeTowerVisionConfig,
    )
    from .processing_bridgetower import BridgeTowerProcessor

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_bridgetower import BridgeTowerImageProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_bridgetower import (
            BridgeTowerForContrastiveLearning,
            BridgeTowerForImageAndTextRetrieval,
            BridgeTowerForMaskedLM,
            BridgeTowerModel,
            BridgeTowerPreTrainedModel,
        )


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
