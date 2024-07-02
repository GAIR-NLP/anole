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

from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_tf_available, is_torch_available


_import_structure = {
    "configuration_data2vec_audio": ["Data2VecAudioConfig"],
    "configuration_data2vec_text": [
        "Data2VecTextConfig",
        "Data2VecTextOnnxConfig",
    ],
    "configuration_data2vec_vision": [
        "Data2VecVisionConfig",
        "Data2VecVisionOnnxConfig",
    ],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_data2vec_audio"] = [
        "Data2VecAudioForAudioFrameClassification",
        "Data2VecAudioForCTC",
        "Data2VecAudioForSequenceClassification",
        "Data2VecAudioForXVector",
        "Data2VecAudioModel",
        "Data2VecAudioPreTrainedModel",
    ]
    _import_structure["modeling_data2vec_text"] = [
        "Data2VecTextForCausalLM",
        "Data2VecTextForMaskedLM",
        "Data2VecTextForMultipleChoice",
        "Data2VecTextForQuestionAnswering",
        "Data2VecTextForSequenceClassification",
        "Data2VecTextForTokenClassification",
        "Data2VecTextModel",
        "Data2VecTextPreTrainedModel",
    ]
    _import_structure["modeling_data2vec_vision"] = [
        "Data2VecVisionForImageClassification",
        "Data2VecVisionForMaskedImageModeling",
        "Data2VecVisionForSemanticSegmentation",
        "Data2VecVisionModel",
        "Data2VecVisionPreTrainedModel",
    ]

if is_tf_available():
    _import_structure["modeling_tf_data2vec_vision"] = [
        "TFData2VecVisionForImageClassification",
        "TFData2VecVisionForSemanticSegmentation",
        "TFData2VecVisionModel",
        "TFData2VecVisionPreTrainedModel",
    ]

if TYPE_CHECKING:
    from .configuration_data2vec_audio import Data2VecAudioConfig
    from .configuration_data2vec_text import (
        Data2VecTextConfig,
        Data2VecTextOnnxConfig,
    )
    from .configuration_data2vec_vision import (
        Data2VecVisionConfig,
        Data2VecVisionOnnxConfig,
    )

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_data2vec_audio import (
            Data2VecAudioForAudioFrameClassification,
            Data2VecAudioForCTC,
            Data2VecAudioForSequenceClassification,
            Data2VecAudioForXVector,
            Data2VecAudioModel,
            Data2VecAudioPreTrainedModel,
        )
        from .modeling_data2vec_text import (
            Data2VecTextForCausalLM,
            Data2VecTextForMaskedLM,
            Data2VecTextForMultipleChoice,
            Data2VecTextForQuestionAnswering,
            Data2VecTextForSequenceClassification,
            Data2VecTextForTokenClassification,
            Data2VecTextModel,
            Data2VecTextPreTrainedModel,
        )
        from .modeling_data2vec_vision import (
            Data2VecVisionForImageClassification,
            Data2VecVisionForMaskedImageModeling,
            Data2VecVisionForSemanticSegmentation,
            Data2VecVisionModel,
            Data2VecVisionPreTrainedModel,
        )
    if is_tf_available():
        from .modeling_tf_data2vec_vision import (
            TFData2VecVisionForImageClassification,
            TFData2VecVisionForSemanticSegmentation,
            TFData2VecVisionModel,
            TFData2VecVisionPreTrainedModel,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
