"""
 Copyright (C) 2024 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

from model_api.adapters.utils import RESIZE_TYPES, InputTransform
from .model import Model
from .utils import ClassificationResult
from .types import BooleanValue, ListValue, NumericalValue, StringValue

if TYPE_CHECKING:
    from model_api.adapters.inference_adapter import InferenceAdapter


class ActionClassificationModel(Model):
    """An wrapper for an action classification model

    The ActionClassificationModel has 1 or more inputs with images - 6D tensors.
    It may support additional inputs - 4D tensors.

    Video format is different from image format, so OpenVINO PrePostProcessors isn't available.
    For that reason, basic preprocessing such as resize and normalize are conducted in this class.

    Attributes:
        image_blob_names (List[str]): names of all image-like inputs (4D tensors)
        image_info_blob_names (List[str]): names of all secondary inputs (2D tensors)
        image_blob_name (str): name of the first image input
        resize_type (str): the type for image resizing (see `RESIZE_TYPE` for info)
        resize (function): resizing function corresponding to the `resize_type`
        input_transform (InputTransform): instance of the `InputTransform` for image normalization
    """

    __model__ = "Action Classification"

    def __init__(
        self,
        inference_adapter: InferenceAdapter,
        configuration: dict[str, Any] =dict(),
        preload: bool =False
    ) -> None:
        """Action classaification model constructor

        Args:
            inference_adapter (InferenceAdapter): allows working with the specified executor
            configuration (dict, optional): it contains values for parameters accepted by specific
              wrapper (`confidence_threshold`, `labels` etc.) which are set as data attributes
            preload (bool, optional): a flag whether the model is loaded to device while
              initialization. If `preload=False`, the model must be loaded via `load` method before inference

        Raises:
            WrapperError: if the wrapper failed to define appropriate inputs for images
        """
        super().__init__(inference_adapter, configuration, preload)
        self.image_blob_names, self.image_info_blob_names = self._get_inputs()
        self.image_blob_name = self.image_blob_names[0]
        self.ncthw_layout = "NCTHW" in self.inputs[self.image_blob_name].layout
        if self.ncthw_layout:
            _, self.n, self.c, self.t, self.h, self.w = self.inputs[self.image_blob_name].shape
        else:
            _, self.n, self.t, self.h, self.w, self.c = self.inputs[self.image_blob_name].shape
        self.resize = RESIZE_TYPES[self.resize_type]
        self.input_transform = InputTransform(
            self.reverse_input_channels, self.mean_values, self.scale_values
        )

    @classmethod
    def parameters(cls) -> dict[str, Any]:
        parameters = super().parameters()
        parameters.update(
            {
                "mean_values": ListValue(
                    description="Normalization values, which will be subtracted from image channels for image-input layer during preprocessing",
                    default_value=[],
                ),
                "pad_value": NumericalValue(
                    int,
                    min=0,
                    max=255,
                    description="Pad value for resize_image_letterbox embedded into a model",
                    default_value=0,
                ),
                "resize_type": StringValue(
                    default_value="standard",
                    choices=tuple(RESIZE_TYPES.keys()),
                    description="Type of input image resizing",
                ),
                "reverse_input_channels": BooleanValue(
                    default_value=False, description="Reverse the input channel order"
                ),
                "scale_values": ListValue(
                    default_value=[],
                    description="Normalization values, which will divide the image channels for image-input layer",
                ),
            }
        )
        return parameters

    def _get_inputs(self) -> tuple[list, list]:
        """Defines the model inputs for images and additional info.

        Raises:
            WrapperError: if the wrapper failed to define appropriate inputs for images

        Returns:
            - list of inputs names for images
            - list of inputs names for additional info
        """
        image_blob_names, image_info_blob_names = [], []
        for name, metadata in self.inputs.items():
            if len(metadata.shape) == 6:
                image_blob_names.append(name)
            elif len(metadata.shape) == 4:
                image_info_blob_names.append(name)
            else:
                self.raise_error(
                    "Failed to identify the input for ImageModel: only 4D and 6D input layer supported"
                )
        if not image_blob_names:
            self.raise_error(
                "Failed to identify the input for the image: no 6D input layer found"
            )
        return image_blob_names, image_info_blob_names

    def preprocess(self, inputs: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, tuple[int, ...]]]:
        """Data preprocess method

        It performs basic preprocessing of a single image:
            - Resizes the image to fit the model input size via the defined resize type
            - Normalizes the image: subtracts means, divides by scales, switch channels BGR-RGB
            - Changes the image layout according to the model input layout

        Also, it keeps the size of original image and resized one as `original_shape` and `resized_shape`
        in the metadata dictionary.

        Note:
            It supports only models with single image input. If the model has more image inputs or has
            additional supported inputs, the `preprocess` should be overloaded in a specific wrapper.

        Args:
            inputs (ndarray): a single image as 3D array in HWC layout

        Returns:
            - the preprocessed image in the following format:
                {
                    'input_layer_name': preprocessed_image
                }
            - the input metadata, which might be used in `postprocess` method
        """
        meta = {"original_shape": inputs.shape, "resized_shape": (1, self.n, self.c, self.t, self.h, self.w)}
        resized_inputs = [self.resize(frame, (self.w, self.h), pad_value=self.pad_value) for frame in inputs]
        frames = self.input_transform(np.array(resized_inputs))
        np_frames = self._change_layout(frames)
        dict_inputs = {self.image_blob_name: np_frames}
        return dict_inputs, meta

    def _change_layout(self, inputs: list[np.ndarray]) -> np.ndarray:
        """Reshape(expand, transpose, permute) the input np.ndarray."""
        np_inputs = np.expand_dims(inputs, axis=(0, 1))  # [1, 1, T, H, W, C]
        if self.ncthw_layout:
            return np_inputs.transpose(0, 1, -1, 2, 3, 4)  # [1, 1, C, T, H, W]
        return np_inputs

    def postprocess(self, outputs: dict[str, np.ndarray], meta: dict[str, Any]) -> np.ndarray:
        """Post-process."""
        logits = next(iter(outputs.values())).squeeze()
        return get_multiclass_predictions(logits)


def get_multiclass_predictions(logits: np.ndarray) -> ClassificationResult:
    """Get multiclass predictions."""
    index = np.argmax(logits)
    return ClassificationResult([(index, index, logits[index])], np.ndarray(0), np.ndarray(0), np.ndarray(0))
