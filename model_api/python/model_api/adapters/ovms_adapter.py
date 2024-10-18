"""
 Copyright (c) 2021-2024 Intel Corporation

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

import re

import numpy as np

from .inference_adapter import InferenceAdapter, Metadata
from .utils import Layout, get_rt_info_from_dict


class OVMSAdapter(InferenceAdapter):
    """
    Class that allows working with models served by the OpenVINO Model Server
    """

    def __init__(self, target_model: str):
        """Expected format: <address>:<port>/models/<model_name>[:<model_version>]"""
        import tritonclient.http as httpclient

        service_url, self.model_name, self.model_version = _parse_model_arg(
            target_model
        )
        self.client = httpclient.InferenceServerClient(service_url)
        if not self.client.is_model_ready(self.model_name, self.model_version):
            raise RuntimeError(
                f"Requested model: {self.model_name}, version: {self.model_version} is not accessible"
            )

        self.metadata = self.client.get_model_metadata(
            model_name=self.model_name, model_version=self.model_version
        )
        self.inputs = self.get_input_layers()

    def get_input_layers(self):
        return {
            meta["name"]: Metadata(
                {meta["name"]},
                meta["shape"],
                Layout.from_shape(meta["shape"]),
                meta["datatype"],
            )
            for meta in self.metadata["inputs"]
        }

    def get_output_layers(self):
        return {
            meta["name"]: Metadata(
                {meta["name"]},
                shape=meta["shape"],
                precision=meta["datatype"],
            )
            for meta in self.metadata["outputs"]
        }

    def infer_sync(self, dict_data):
        inputs = _prepare_inputs(dict_data, self.inputs)
        raw_result = self.client.infer(
            model_name=self.model_name, model_version=self.model_version, inputs=inputs
        )

        inference_results = {}
        for output in self.metadata["outputs"]:
            inference_results[output["name"]] = raw_result.as_numpy(output["name"])

        return inference_results

    def infer_async(self, dict_data, callback_data):
        inputs = _prepare_inputs(dict_data, self.inputs)
        raw_result = self.client.infer(
            model_name=self.model_name, model_version=self.model_version, inputs=inputs
        )
        inference_results = {}
        for output in self.metadata["outputs"]:
            inference_results[output["name"]] = raw_result.as_numpy(output["name"])

        self.callback_fn(inference_results, (lambda x: x, callback_data))

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
        pass

    def reshape_model(self, new_shape):
        raise NotImplementedError

    def get_rt_info(self, path):
        return get_rt_info_from_dict(self.metadata["rt_info"], path)


_triton2np_precision = {
    "INT64": np.int64,
    "UINT64": np.uint64,
    "FLOAT": np.float32,
    "UINT32": np.uint32,
    "INT32": np.int32,
    "HALF": np.float16,
    "INT16": np.int16,
    "INT8": np.int8,
    "UINT8": np.uint8,
}


def _parse_model_arg(target_model: str):
    if not isinstance(target_model, str):
        raise TypeError("target_model must be str")
    # Expected format: <address>:<port>/models/<model_name>[:<model_version>]
    if not re.fullmatch(
        r"(\w+\.*\-*)*\w+:\d+\/models\/[a-zA-Z0-9._-]+(\:\d+)*", target_model
    ):
        raise ValueError("invalid --model option format")
    service_url, _, model = target_model.split("/")
    model_spec = model.split(":")
    if len(model_spec) == 1:
        # model version not specified - use latest
        return service_url, model_spec[0], ""
    if len(model_spec) == 2:
        return service_url, model_spec[0], model_spec[1]
    raise ValueError("invalid target_model format")


def _prepare_inputs(dict_data, inputs_meta):
    import tritonclient.http as httpclient

    inputs = []
    for input_name, input_data in dict_data.items():
        if input_name not in inputs_meta.keys():
            raise ValueError("Input data does not match model inputs")
        input_info = inputs_meta[input_name]
        model_precision = _triton2np_precision[input_info.precision]
        if isinstance(input_data, np.ndarray) and input_data.dtype != model_precision:
            input_data = input_data.astype(model_precision)
        elif isinstance(input_data, list):
            input_data = np.array(input_data, dtype=model_precision)

        infer_input = httpclient.InferInput(
            input_name, input_data.shape, input_info.precision
        )
        infer_input.set_data_from_numpy(input_data)
        inputs.append(infer_input)
    return inputs
