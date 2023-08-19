"""
 Copyright (c) 2023 Intel Corporation

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

from functools import partial, reduce
import numpy as np
from openvino.model_api.models.utils import INTERPOLATION_TYPES, RESIZE_TYPES, InputTransform

try:
    import onnxruntime as ort
    import onnx
    onnxrt_absent = False
except ImportError:
    onnxrt_absent = True

from .inference_adapter import InferenceAdapter, Metadata
from .utils import Layout


class ParamWrapper:
    def __init__(self, value):
        self.value = value

    def astype(self, type):
        return type(self.value)


class ONNXRuntimeAdapter(InferenceAdapter):
    """
    Class that allows running ONNX models via ONNX backend
    """

    def __init__(self, model_path: str):
        """"""
        self.session = ort.InferenceSession(model_path)
        self.output_names = [o.name for o in self.session.get_outputs()]

        self.onnx_metadata = {}
        onnx_model = onnx.load(model_path)
        def insert_hiararchical(keys, val, root_dict):
            if len(keys) == 1:
                root_dict[keys[0]] = val
                return
            if keys[0] not in root_dict:
                root_dict[keys[0]] = {}
            insert_hiararchical(keys[1:], val, root_dict[keys[0]])

        for prop in onnx_model.metadata_props:
            keys = prop.key.split()
            if "model_info" in keys:
                insert_hiararchical(keys, prop.value, self.onnx_metadata)

        self.preprocessor = lambda arg: arg

    def get_input_layers(self):
        inputs = {}

        for input in self.session.get_inputs():
            shape = get_shape_from_onnx(input.shape)
            inputs[input.name] = Metadata(
                {input.name},
                shape,
                Layout.from_shape(shape),
                _onnx2ov_precision.get(input.type, input.type))

        return inputs

    def get_output_layers(self):
        outputs = {}
        for output in self.session.get_outputs():
            shape = get_shape_from_onnx(output.shape)
            outputs[output.name] = Metadata(
                {output.name},
                shape=shape,
                precision=_onnx2ov_precision.get(output.type, output.type))

        return outputs

    def infer_sync(self, dict_data):
        inputs = {}
        for input in self.session.get_inputs():
            preprocessed_input = self.preprocessor(dict_data[input.name])
            if dict_data[input.name].dtype != _onnx2np_precision[input.type]:
                inputs[input.name] = ort.OrtValue.ortvalue_from_numpy(preprocessed_input.astype(_onnx2np_precision[input.type]))
            else:
                inputs[input.name] = ort.OrtValue.ortvalue_from_numpy(preprocessed_input)
        raw_result = self.session.run(
            self.output_names, inputs)

        named_raw_result = {}
        for i, data in enumerate(raw_result):
            named_raw_result[self.output_names[i]] = data

        return named_raw_result

    def infer_async(self, dict_data, callback_data):
        raise NotImplementedError

    def set_callback(self, callback_fn):
        self.callback_fn = callback_fn

    def is_ready(self):
        return True

    def load_model(self):
        pass

    def await_all(self):
        pass

    def await_any(self):
        pass

    def embed_preprocessing(
        self,
        layout,
        resize_mode: str,
        interpolation_mode,
        target_shape,
        pad_value,
        dtype=type(int),
        brg2rgb=False,
        mean=None,
        scale=None,
        input_idx=0,
    ):
        preproc_funcs = [np.squeeze]
        if resize_mode != "crop":
            resize_fn = partial(RESIZE_TYPES[resize_mode], size=target_shape, interpolation=INTERPOLATION_TYPES[interpolation_mode])
        else:
            resize_fn = partial(RESIZE_TYPES[resize_mode], size=target_shape)
        preproc_funcs.append(resize_fn)
        input_transform = InputTransform(
            brg2rgb, mean, scale
        )
        preproc_funcs.append(input_transform.__call__)
        preproc_funcs.append(change_layout)

        self.preprocessor = reduce(lambda f, g: lambda x: f(g(x)), reversed(preproc_funcs))

    def reshape_model(self, new_shape):
        raise NotImplementedError

    def get_rt_info(self, path):
        value = self.onnx_metadata
        try:
            value = self.onnx_metadata
            for item in path:
                value = value[item]
            return ParamWrapper(value)
        except KeyError:
            raise RuntimeError(
                "Cannot get runtime attribute. Path to runtime attribute is incorrect."
            )


_onnx2ov_precision = {
    "tensor(float)": "f32",
}

_onnx2np_precision = {
    "tensor(float)": np.float32,
}

def get_shape_from_onnx(onnx_shape):
    for i, item in enumerate(onnx_shape):
        if isinstance(item, str):
            onnx_shape[i] = -1
    return tuple(onnx_shape)

def change_layout(image):
    """Changes the input image layout to fit the layout of the model input layer.

    Args:
        inputs (ndarray): a single image as 3D array in HWC layout

    Returns:
        - the image with layout aligned with the model layout
    if self.nchw_layout:
        image = image.transpose((2, 0, 1))  # HWC->CHW
        image = image.reshape((1, self.c, self.h, self.w))
    return image
    """
    image = image.transpose((2, 0, 1))  # HWC->CHW
    image = image.reshape((1, *image.shape))
    return image
