#!/usr/bin/env python
# coding=utf-8

# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
import base64
import importlib
import io
import json
import os
import tempfile
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

from huggingface_hub import create_repo, get_collection, hf_hub_download, metadata_update, upload_folder
from huggingface_hub.utils import RepositoryNotFoundError, build_hf_headers, get_session
from packaging import version

from ..dynamic_module_utils import (
    custom_object_save,
    get_class_from_dynamic_module,
    get_imports,
)
from ..models.auto import AutoProcessor
from ..utils import (
    CONFIG_NAME,
    cached_file,
    is_accelerate_available,
    is_torch_available,
    is_vision_available,
    logging,
)
from .agent_types import handle_agent_inputs, handle_agent_outputs


logger = logging.get_logger(__name__)


if is_torch_available():
    import torch

if is_accelerate_available():
    from accelerate import PartialState
    from accelerate.utils import send_to_device


TOOL_CONFIG_FILE = "tool_config.json"


def get_repo_type(repo_id, repo_type=None, **hub_kwargs):
    if repo_type is not None:
        return repo_type
    try:
        hf_hub_download(repo_id, TOOL_CONFIG_FILE, repo_type="space", **hub_kwargs)
        return "space"
    except RepositoryNotFoundError:
        try:
            hf_hub_download(repo_id, TOOL_CONFIG_FILE, repo_type="model", **hub_kwargs)
            return "model"
        except RepositoryNotFoundError:
            raise EnvironmentError(f"`{repo_id}` does not seem to be a valid repo identifier on the Hub.")
        except Exception:
            return "model"
    except Exception:
        return "space"


# docstyle-ignore
APP_FILE_TEMPLATE = """from transformers import launch_gradio_demo
from {module_name} import {class_name}

launch_gradio_demo({class_name})
"""


class Tool:
    """
    A base class for the functions used by the agent. Subclass this and implement the `__call__` method as well as the
    following class attributes:

    - **description** (`str`) -- A short description of what your tool does, the inputs it expects and the output(s) it
      will return. For instance 'This is a tool that downloads a file from a `url`. It takes the `url` as input, and
      returns the text contained in the file'.
    - **name** (`str`) -- A performative name that will be used for your tool in the prompt to the agent. For instance
      `"text-classifier"` or `"image_generator"`.
    - **inputs** (`Dict[str, Dict[str, Union[str, type]]]`) -- The dict of modalities expected for the inputs.
      It has one `type`key and a `description`key.
      This is used by `launch_gradio_demo` or to make a nice space from your tool, and also can be used in the generated
      description for your tool.
    - **output_type** (`type`) -- The type of the tool output. This is used by `launch_gradio_demo`
      or to make a nice space from your tool, and also can be used in the generated description for your tool.

    You can also override the method [`~Tool.setup`] if your tool as an expensive operation to perform before being
    usable (such as loading a model). [`~Tool.setup`] will be called the first time you use your tool, but not at
    instantiation.
    """

    name: str
    description: str
    inputs: Dict[str, Dict[str, Union[str, type]]]
    output_type: type

    def __init__(self, *args, **kwargs):
        self.is_initialized = False

    def validate_attributes(self):
        required_attributes = {
            "description": str,
            "name": str,
            "inputs": Dict,
            "output_type": type,
        }
        for attr, expected_type in required_attributes.items():
            attr_value = getattr(self, attr, None)
            if not isinstance(attr_value, expected_type):
                raise TypeError(f"Instance attribute {attr} must exist and be of type {expected_type.__name__}")

    def forward(self, *args, **kwargs):
        return NotImplemented("Write this method in your subclass of `Tool`.")

    def __call__(self, *args, **kwargs):
        args, kwargs = handle_agent_inputs(*args, **kwargs)
        outputs = self.forward(*args, **kwargs)
        return handle_agent_outputs(outputs, self.output_type)

    def setup(self):
        """
        Overwrite this method here for any operation that is expensive and needs to be executed before you start using
        your tool. Such as loading a big model.
        """
        self.is_initialized = True

    def save(self, output_dir):
        """
        Saves the relevant code files for your tool so it can be pushed to the Hub. This will copy the code of your
        tool in `output_dir` as well as autogenerate:

        - a config file named `tool_config.json`
        - an `app.py` file so that your tool can be converted to a space
        - a `requirements.txt` containing the names of the module used by your tool (as detected when inspecting its
          code)

        You should only use this method to save tools that are defined in a separate module (not `__main__`).

        Args:
            output_dir (`str`): The folder in which you want to save your tool.
        """
        os.makedirs(output_dir, exist_ok=True)
        # Save module file
        if self.__module__ == "__main__":
            raise ValueError(
                f"We can't save the code defining {self} in {output_dir} as it's been defined in __main__. You "
                "have to put this code in a separate module so we can include it in the saved folder."
            )
        module_files = custom_object_save(self, output_dir)

        module_name = self.__class__.__module__
        last_module = module_name.split(".")[-1]
        full_name = f"{last_module}.{self.__class__.__name__}"

        # Save config file
        config_file = os.path.join(output_dir, "tool_config.json")
        if os.path.isfile(config_file):
            with open(config_file, "r", encoding="utf-8") as f:
                tool_config = json.load(f)
        else:
            tool_config = {}

        tool_config = {
            "tool_class": full_name,
            "description": self.description,
            "name": self.name,
            "inputs": self.inputs,
            "output_type": str(self.output_type),
        }
        with open(config_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(tool_config, indent=2, sort_keys=True) + "\n")

        # Save app file
        app_file = os.path.join(output_dir, "app.py")
        with open(app_file, "w", encoding="utf-8") as f:
            f.write(APP_FILE_TEMPLATE.format(module_name=last_module, class_name=self.__class__.__name__))

        # Save requirements file
        requirements_file = os.path.join(output_dir, "requirements.txt")
        imports = []
        for module in module_files:
            imports.extend(get_imports(module))
        imports = list(set(imports))
        with open(requirements_file, "w", encoding="utf-8") as f:
            f.write("\n".join(imports) + "\n")

    @classmethod
    def from_hub(
        cls,
        repo_id: str,
        model_repo_id: Optional[str] = None,
        token: Optional[str] = None,
        **kwargs,
    ):
        """
        Loads a tool defined on the Hub.

        <Tip warning={true}>

        Loading a tool from the Hub means that you'll download the tool and execute it locally.
        ALWAYS inspect the tool you're downloading before loading it within your runtime, as you would do when
        installing a package using pip/npm/apt.

        </Tip>

        Args:
            repo_id (`str`):
                The name of the repo on the Hub where your tool is defined.
            model_repo_id (`str`, *optional*):
                If your tool uses a model and you want to use a different model than the default, you can pass a second
                repo ID or an endpoint url to this argument.
            token (`str`, *optional*):
                The token to identify you on hf.co. If unset, will use the token generated when running
                `huggingface-cli login` (stored in `~/.huggingface`).
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments that will be split in two: all arguments relevant to the Hub (such as
                `cache_dir`, `revision`, `subfolder`) will be used when downloading the files for your tool, and the
                others will be passed along to its init.
        """
        hub_kwargs_names = [
            "cache_dir",
            "force_download",
            "resume_download",
            "proxies",
            "revision",
            "repo_type",
            "subfolder",
            "local_files_only",
        ]
        hub_kwargs = {k: v for k, v in kwargs.items() if k in hub_kwargs_names}

        # Try to get the tool config first.
        hub_kwargs["repo_type"] = get_repo_type(repo_id, **hub_kwargs)
        resolved_config_file = cached_file(
            repo_id,
            TOOL_CONFIG_FILE,
            token=token,
            **hub_kwargs,
            _raise_exceptions_for_gated_repo=False,
            _raise_exceptions_for_missing_entries=False,
            _raise_exceptions_for_connection_errors=False,
        )
        is_tool_config = resolved_config_file is not None
        if resolved_config_file is None:
            resolved_config_file = cached_file(
                repo_id,
                CONFIG_NAME,
                token=token,
                **hub_kwargs,
                _raise_exceptions_for_gated_repo=False,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
            )
        if resolved_config_file is None:
            raise EnvironmentError(
                f"{repo_id} does not appear to provide a valid configuration in `tool_config.json` or `config.json`."
            )

        with open(resolved_config_file, encoding="utf-8") as reader:
            config = json.load(reader)

        if not is_tool_config:
            if "custom_tool" not in config:
                raise EnvironmentError(
                    f"{repo_id} does not provide a mapping to custom tools in its configuration `config.json`."
                )
            custom_tool = config["custom_tool"]
        else:
            custom_tool = config

        tool_class = custom_tool["tool_class"]
        tool_class = get_class_from_dynamic_module(tool_class, repo_id, token=token, **hub_kwargs)

        if len(tool_class.name) == 0:
            tool_class.name = custom_tool["name"]
        if tool_class.name != custom_tool["name"]:
            logger.warning(
                f"{tool_class.__name__} implements a different name in its configuration and class. Using the tool "
                "configuration name."
            )
            tool_class.name = custom_tool["name"]

        if len(tool_class.description) == 0:
            tool_class.description = custom_tool["description"]
        if tool_class.description != custom_tool["description"]:
            logger.warning(
                f"{tool_class.__name__} implements a different description in its configuration and class. Using the "
                "tool configuration description."
            )
            tool_class.description = custom_tool["description"]

        if tool_class.inputs != custom_tool["inputs"]:
            tool_class.inputs = custom_tool["inputs"]
        if tool_class.output_type != custom_tool["output_type"]:
            tool_class.output_type = custom_tool["output_type"]

        return tool_class(**kwargs)

    def push_to_hub(
        self,
        repo_id: str,
        commit_message: str = "Upload tool",
        private: Optional[bool] = None,
        token: Optional[Union[bool, str]] = None,
        create_pr: bool = False,
    ) -> str:
        """
        Upload the tool to the Hub.

        For this method to work properly, your tool must have been defined in a separate module (not `__main__`).
        For instance:
        ```
        from my_tool_module import MyTool
        my_tool = MyTool()
        my_tool.push_to_hub("my-username/my-space")
        ```

        Parameters:
            repo_id (`str`):
                The name of the repository you want to push your tool to. It should contain your organization name when
                pushing to a given organization.
            commit_message (`str`, *optional*, defaults to `"Upload tool"`):
                Message to commit while pushing.
            private (`bool`, *optional*):
                Whether or not the repository created should be private.
            token (`bool` or `str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If unset, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`).
            create_pr (`bool`, *optional*, defaults to `False`):
                Whether or not to create a PR with the uploaded files or directly commit.
        """
        repo_url = create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            exist_ok=True,
            repo_type="space",
            space_sdk="gradio",
        )
        repo_id = repo_url.repo_id
        metadata_update(repo_id, {"tags": ["tool"]}, repo_type="space")

        with tempfile.TemporaryDirectory() as work_dir:
            # Save all files.
            self.save(work_dir)
            logger.info(f"Uploading the following files to {repo_id}: {','.join(os.listdir(work_dir))}")
            return upload_folder(
                repo_id=repo_id,
                commit_message=commit_message,
                folder_path=work_dir,
                token=token,
                create_pr=create_pr,
                repo_type="space",
            )

    @staticmethod
    def from_gradio(gradio_tool):
        """
        Creates a [`Tool`] from a gradio tool.
        """
        import inspect

        class GradioToolWrapper(Tool):
            def __init__(self, _gradio_tool):
                super().__init__()
                self.name = _gradio_tool.name
                self.description = _gradio_tool.description
                self.output_type = "text"
                self._gradio_tool = _gradio_tool
                func_args = list(inspect.signature(_gradio_tool.run).parameters.keys())
                self.inputs = {key: "" for key in func_args}

            def forward(self, *args, **kwargs):
                return self._gradio_tool.run(*args, **kwargs)

        return GradioToolWrapper(gradio_tool)

    @staticmethod
    def from_langchain(langchain_tool):
        """
        Creates a [`Tool`] from a langchain tool.
        """

        class LangChainToolWrapper(Tool):
            def __init__(self, _langchain_tool):
                super().__init__()
                self.name = _langchain_tool.name.lower()
                self.description = _langchain_tool.description
                self.inputs = parse_langchain_args(_langchain_tool.args)
                self.output_type = "text"
                self.langchain_tool = _langchain_tool

            def forward(self, *args, **kwargs):
                tool_input = kwargs.copy()
                for index, argument in enumerate(args):
                    if index < len(self.inputs):
                        input_key = next(iter(self.inputs))
                        tool_input[input_key] = argument
                return self.langchain_tool.run(tool_input)

        return LangChainToolWrapper(langchain_tool)


DEFAULT_TOOL_DESCRIPTION_TEMPLATE = """
- {{ tool.name }}: {{ tool.description }}
    Takes inputs: {{tool.inputs}}
"""


def get_tool_description_with_args(tool: Tool, description_template: str = DEFAULT_TOOL_DESCRIPTION_TEMPLATE) -> str:
    compiled_template = compile_jinja_template(description_template)
    rendered = compiled_template.render(
        tool=tool,
    )
    return rendered


@lru_cache
def compile_jinja_template(template):
    try:
        import jinja2
        from jinja2.exceptions import TemplateError
        from jinja2.sandbox import ImmutableSandboxedEnvironment
    except ImportError:
        raise ImportError("template requires jinja2 to be installed.")

    if version.parse(jinja2.__version__) < version.parse("3.1.0"):
        raise ImportError("template requires jinja2>=3.1.0 to be installed. Your version is " f"{jinja2.__version__}.")

    def raise_exception(message):
        raise TemplateError(message)

    jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    jinja_env.globals["raise_exception"] = raise_exception
    return jinja_env.from_string(template)


class PipelineTool(Tool):
    """
    A [`Tool`] tailored towards Transformer models. On top of the class attributes of the base class [`Tool`], you will
    need to specify:

    - **model_class** (`type`) -- The class to use to load the model in this tool.
    - **default_checkpoint** (`str`) -- The default checkpoint that should be used when the user doesn't specify one.
    - **pre_processor_class** (`type`, *optional*, defaults to [`AutoProcessor`]) -- The class to use to load the
      pre-processor
    - **post_processor_class** (`type`, *optional*, defaults to [`AutoProcessor`]) -- The class to use to load the
      post-processor (when different from the pre-processor).

    Args:
        model (`str` or [`PreTrainedModel`], *optional*):
            The name of the checkpoint to use for the model, or the instantiated model. If unset, will default to the
            value of the class attribute `default_checkpoint`.
        pre_processor (`str` or `Any`, *optional*):
            The name of the checkpoint to use for the pre-processor, or the instantiated pre-processor (can be a
            tokenizer, an image processor, a feature extractor or a processor). Will default to the value of `model` if
            unset.
        post_processor (`str` or `Any`, *optional*):
            The name of the checkpoint to use for the post-processor, or the instantiated pre-processor (can be a
            tokenizer, an image processor, a feature extractor or a processor). Will default to the `pre_processor` if
            unset.
        device (`int`, `str` or `torch.device`, *optional*):
            The device on which to execute the model. Will default to any accelerator available (GPU, MPS etc...), the
            CPU otherwise.
        device_map (`str` or `dict`, *optional*):
            If passed along, will be used to instantiate the model.
        model_kwargs (`dict`, *optional*):
            Any keyword argument to send to the model instantiation.
        token (`str`, *optional*):
            The token to use as HTTP bearer authorization for remote files. If unset, will use the token generated when
            running `huggingface-cli login` (stored in `~/.huggingface`).
        hub_kwargs (additional keyword arguments, *optional*):
            Any additional keyword argument to send to the methods that will load the data from the Hub.
    """

    pre_processor_class = AutoProcessor
    model_class = None
    post_processor_class = AutoProcessor
    default_checkpoint = None
    description = "This is a pipeline tool"
    name = "pipeline"
    inputs = {"prompt": str}
    output_type = str

    def __init__(
        self,
        model=None,
        pre_processor=None,
        post_processor=None,
        device=None,
        device_map=None,
        model_kwargs=None,
        token=None,
        **hub_kwargs,
    ):
        if not is_torch_available():
            raise ImportError("Please install torch in order to use this tool.")

        if not is_accelerate_available():
            raise ImportError("Please install accelerate in order to use this tool.")

        if model is None:
            if self.default_checkpoint is None:
                raise ValueError("This tool does not implement a default checkpoint, you need to pass one.")
            model = self.default_checkpoint
        if pre_processor is None:
            pre_processor = model

        self.model = model
        self.pre_processor = pre_processor
        self.post_processor = post_processor
        self.device = device
        self.device_map = device_map
        self.model_kwargs = {} if model_kwargs is None else model_kwargs
        if device_map is not None:
            self.model_kwargs["device_map"] = device_map
        self.hub_kwargs = hub_kwargs
        self.hub_kwargs["token"] = token

        super().__init__()

    def setup(self):
        """
        Instantiates the `pre_processor`, `model` and `post_processor` if necessary.
        """
        if isinstance(self.pre_processor, str):
            self.pre_processor = self.pre_processor_class.from_pretrained(self.pre_processor, **self.hub_kwargs)

        if isinstance(self.model, str):
            self.model = self.model_class.from_pretrained(self.model, **self.model_kwargs, **self.hub_kwargs)

        if self.post_processor is None:
            self.post_processor = self.pre_processor
        elif isinstance(self.post_processor, str):
            self.post_processor = self.post_processor_class.from_pretrained(self.post_processor, **self.hub_kwargs)

        if self.device is None:
            if self.device_map is not None:
                self.device = list(self.model.hf_device_map.values())[0]
            else:
                self.device = PartialState().default_device

        if self.device_map is None:
            self.model.to(self.device)

        super().setup()

    def encode(self, raw_inputs):
        """
        Uses the `pre_processor` to prepare the inputs for the `model`.
        """
        return self.pre_processor(raw_inputs)

    def forward(self, inputs):
        """
        Sends the inputs through the `model`.
        """
        with torch.no_grad():
            return self.model(**inputs)

    def decode(self, outputs):
        """
        Uses the `post_processor` to decode the model output.
        """
        return self.post_processor(outputs)

    def __call__(self, *args, **kwargs):
        args, kwargs = handle_agent_inputs(*args, **kwargs)

        if not self.is_initialized:
            self.setup()

        encoded_inputs = self.encode(*args, **kwargs)

        tensor_inputs = {k: v for k, v in encoded_inputs.items() if isinstance(v, torch.Tensor)}
        non_tensor_inputs = {k: v for k, v in encoded_inputs.items() if not isinstance(v, torch.Tensor)}

        encoded_inputs = send_to_device(tensor_inputs, self.device)
        outputs = self.forward({**encoded_inputs, **non_tensor_inputs})
        outputs = send_to_device(outputs, "cpu")
        decoded_outputs = self.decode(outputs)

        return handle_agent_outputs(decoded_outputs, self.output_type)


def launch_gradio_demo(tool_class: Tool):
    """
    Launches a gradio demo for a tool. The corresponding tool class needs to properly implement the class attributes
    `inputs` and `output_type`.

    Args:
        tool_class (`type`): The class of the tool for which to launch the demo.
    """
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("Gradio should be installed in order to launch a gradio demo.")

    tool = tool_class()

    def fn(*args, **kwargs):
        return tool(*args, **kwargs)

    gradio_inputs = []
    for input_name, input_details in tool_class.inputs.items():
        input_type = input_details["type"]
        if input_type == "text":
            gradio_inputs.append(gr.Textbox(label=input_name))
        elif input_type == "image":
            gradio_inputs.append(gr.Image(label=input_name))
        elif input_type == "audio":
            gradio_inputs.append(gr.Audio(label=input_name))
        else:
            error_message = f"Input type '{input_type}' not supported."
            raise ValueError(error_message)

    gradio_output = tool_class.output_type
    assert gradio_output in ["text", "image", "audio"], f"Output type '{gradio_output}' not supported."

    gr.Interface(
        fn=fn,
        inputs=gradio_inputs,
        outputs=gradio_output,
        title=tool_class.__name__,
        article=tool.description,
    ).launch()


TASK_MAPPING = {
    "document-question-answering": "DocumentQuestionAnsweringTool",
    "image-question-answering": "ImageQuestionAnsweringTool",
    "speech-to-text": "SpeechToTextTool",
    "text-to-speech": "TextToSpeechTool",
    "translation": "TranslationTool",
    "python_interpreter": "PythonInterpreterTool",
    "final_answer": "FinalAnswerTool",
}


def load_tool(task_or_repo_id, model_repo_id=None, token=None, **kwargs):
    """
    Main function to quickly load a tool, be it on the Hub or in the Transformers library.

    <Tip warning={true}>

    Loading a tool means that you'll download the tool and execute it locally.
    ALWAYS inspect the tool you're downloading before loading it within your runtime, as you would do when
    installing a package using pip/npm/apt.

    </Tip>

    Args:
        task_or_repo_id (`str`):
            The task for which to load the tool or a repo ID of a tool on the Hub. Tasks implemented in Transformers
            are:

            - `"document-question-answering"`
            - `"image-question-answering"`
            - `"speech-to-text"`
            - `"text-to-speech"`
            - `"translation"`

        model_repo_id (`str`, *optional*):
            Use this argument to use a different model than the default one for the tool you selected.
        token (`str`, *optional*):
            The token to identify you on hf.co. If unset, will use the token generated when running `huggingface-cli
            login` (stored in `~/.huggingface`).
        kwargs (additional keyword arguments, *optional*):
            Additional keyword arguments that will be split in two: all arguments relevant to the Hub (such as
            `cache_dir`, `revision`, `subfolder`) will be used when downloading the files for your tool, and the others
            will be passed along to its init.
    """
    if task_or_repo_id in TASK_MAPPING:
        tool_class_name = TASK_MAPPING[task_or_repo_id]
        main_module = importlib.import_module("transformers")
        tools_module = main_module.agents
        tool_class = getattr(tools_module, tool_class_name)
        return tool_class(model_repo_id, token=token, **kwargs)
    else:
        logger.warning_once(
            f"You're loading a tool from the Hub from {model_repo_id}. Please make sure this is a source that you "
            f"trust as the code within that tool will be executed on your machine. Always verify the code of "
            f"the tools that you load. We recommend specifying a `revision` to ensure you're loading the "
            f"code that you have checked."
        )
        return Tool.from_hub(task_or_repo_id, model_repo_id=model_repo_id, token=token, **kwargs)


def add_description(description):
    """
    A decorator that adds a description to a function.
    """

    def inner(func):
        func.description = description
        func.name = func.__name__
        return func

    return inner


## Will move to the Hub
class EndpointClient:
    def __init__(self, endpoint_url: str, token: Optional[str] = None):
        self.headers = {
            **build_hf_headers(token=token),
            "Content-Type": "application/json",
        }
        self.endpoint_url = endpoint_url

    @staticmethod
    def encode_image(image):
        _bytes = io.BytesIO()
        image.save(_bytes, format="PNG")
        b64 = base64.b64encode(_bytes.getvalue())
        return b64.decode("utf-8")

    @staticmethod
    def decode_image(raw_image):
        if not is_vision_available():
            raise ImportError(
                "This tool returned an image but Pillow is not installed. Please install it (`pip install Pillow`)."
            )

        from PIL import Image

        b64 = base64.b64decode(raw_image)
        _bytes = io.BytesIO(b64)
        return Image.open(_bytes)

    def __call__(
        self,
        inputs: Optional[Union[str, Dict, List[str], List[List[str]]]] = None,
        params: Optional[Dict] = None,
        data: Optional[bytes] = None,
        output_image: bool = False,
    ) -> Any:
        # Build payload
        payload = {}
        if inputs:
            payload["inputs"] = inputs
        if params:
            payload["parameters"] = params

        # Make API call
        response = get_session().post(self.endpoint_url, headers=self.headers, json=payload, data=data)

        # By default, parse the response for the user.
        if output_image:
            return self.decode_image(response.content)
        else:
            return response.json()


def parse_langchain_args(args: Dict[str, str]) -> Dict[str, str]:
    """Parse the args attribute of a LangChain tool to create a matching inputs dictionary."""
    inputs = args.copy()
    for arg_details in inputs.values():
        if "title" in arg_details:
            arg_details.pop("title")
    return inputs


class ToolCollection:
    """
    Tool collections enable loading all Spaces from a collection in order to be added to the agent's toolbox.

    > [!NOTE]
    > Only Spaces will be fetched, so you can feel free to add models and datasets to your collection if you'd
    > like for this collection to showcase them.

    Args:
        collection_slug (str):
            The collection slug referencing the collection.
        token (str, *optional*):
            The authentication token if the collection is private.

    Example:

    ```py
    >>> from transformers import ToolCollection, ReactCodeAgent

    >>> image_tool_collection = ToolCollection(collection_slug="huggingface-tools/diffusion-tools-6630bb19a942c2306a2cdb6f")
    >>> agent = ReactCodeAgent(tools=[*image_tool_collection.tools], add_base_tools=True)

    >>> agent.run("Please draw me a picture of rivers and lakes.")
    ```
    """

    def __init__(self, collection_slug: str, token: Optional[str] = None):
        self._collection = get_collection(collection_slug, token=token)
        self._hub_repo_ids = {item.item_id for item in self._collection.items if item.item_type == "space"}
        self.tools = {Tool.from_hub(repo_id) for repo_id in self._hub_repo_ids}
