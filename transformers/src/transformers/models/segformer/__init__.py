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
    is_tf_available,
    is_torch_available,
    is_vision_available,
)


_import_structure = {"configuration_segformer": ["SegformerConfig", "SegformerOnnxConfig"]}

try:
    if not is_vision_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["feature_extraction_segformer"] = ["SegformerFeatureExtractor"]
    _import_structure["image_processing_segformer"] = ["SegformerImageProcessor"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_segformer"] = [
        "SegformerDecodeHead",
        "SegformerForImageClassification",
        "SegformerForSemanticSegmentation",
        "SegformerLayer",
        "SegformerModel",
        "SegformerPreTrainedModel",
    ]

try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_tf_segformer"] = [
        "TFSegformerDecodeHead",
        "TFSegformerForImageClassification",
        "TFSegformerForSemanticSegmentation",
        "TFSegformerModel",
        "TFSegformerPreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_segformer import SegformerConfig, SegformerOnnxConfig

    try:
        if not is_vision_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .feature_extraction_segformer import SegformerFeatureExtractor
        from .image_processing_segformer import SegformerImageProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_segformer import (
            SegformerDecodeHead,
            SegformerForImageClassification,
            SegformerForSemanticSegmentation,
            SegformerLayer,
            SegformerModel,
            SegformerPreTrainedModel,
        )
    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_tf_segformer import (
            TFSegformerDecodeHead,
            TFSegformerForImageClassification,
            TFSegformerForSemanticSegmentation,
            TFSegformerModel,
            TFSegformerPreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
