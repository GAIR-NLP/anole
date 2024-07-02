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
from typing import TYPE_CHECKING

from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available, is_vision_available


_import_structure = {
    "configuration_mobilenet_v2": [
        "MobileNetV2Config",
        "MobileNetV2OnnxConfig",
    ],
}

try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["feature_extraction_mobilenet_v2"] = ["MobileNetV2FeatureExtractor"]
    _import_structure["image_processing_mobilenet_v2"] = ["MobileNetV2ImageProcessor"]


try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_mobilenet_v2"] = [
        "MobileNetV2ForImageClassification",
        "MobileNetV2ForSemanticSegmentation",
        "MobileNetV2Model",
        "MobileNetV2PreTrainedModel",
        "load_tf_weights_in_mobilenet_v2",
    ]


if TYPE_CHECKING:
    from .configuration_mobilenet_v2 import (
        MobileNetV2Config,
        MobileNetV2OnnxConfig,
    )

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .feature_extraction_mobilenet_v2 import MobileNetV2FeatureExtractor
        from .image_processing_mobilenet_v2 import MobileNetV2ImageProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_mobilenet_v2 import (
            MobileNetV2ForImageClassification,
            MobileNetV2ForSemanticSegmentation,
            MobileNetV2Model,
            MobileNetV2PreTrainedModel,
            load_tf_weights_in_mobilenet_v2,
        )


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
