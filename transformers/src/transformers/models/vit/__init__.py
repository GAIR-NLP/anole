# Copyright 2021 The HuggingFace Team. All rights reserved.
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

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_torch_available,
    is_torchvision_available,
    is_vision_available,
)


_import_structure = {"configuration_vit": ["ViTConfig", "ViTOnnxConfig"]}

try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["feature_extraction_vit"] = ["ViTFeatureExtractor"]
    _import_structure["image_processing_vit"] = ["ViTImageProcessor"]


try:
    if not is_torchvision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["image_processing_vit_fast"] = ["ViTImageProcessorFast"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_vit"] = [
        "ViTForImageClassification",
        "ViTForMaskedImageModeling",
        "ViTModel",
        "ViTPreTrainedModel",
    ]

try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_vit"] = [
        "TFViTForImageClassification",
        "TFViTModel",
        "TFViTPreTrainedModel",
    ]

try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_flax_vit"] = [
        "FlaxViTForImageClassification",
        "FlaxViTModel",
        "FlaxViTPreTrainedModel",
    ]

if TYPE_CHECKING:
    from .configuration_vit import ViTConfig, ViTOnnxConfig

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .feature_extraction_vit import ViTFeatureExtractor
        from .image_processing_vit import ViTImageProcessor

    try:
        if not is_torchvision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_vit_fast import ViTImageProcessorFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_vit import (
            ViTForImageClassification,
            ViTForMaskedImageModeling,
            ViTModel,
            ViTPreTrainedModel,
        )

    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_vit import TFViTForImageClassification, TFViTModel, TFViTPreTrainedModel

    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_flax_vit import FlaxViTForImageClassification, FlaxViTModel, FlaxViTPreTrainedModel


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
