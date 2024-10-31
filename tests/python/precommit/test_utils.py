#
# Copyright (C) 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import numpy as np
import openvino.runtime as ov
from model_api.adapters.utils import (
    resize_image_with_aspect,
    resize_image_with_aspect_ocv,
)
from openvino.preprocess import PrePostProcessor


def test_resize_image_with_aspect_ocv():
    param_node = ov.op.Parameter(ov.Type.f32, ov.Shape([1, 8, 8, 3]))
    model = ov.Model(param_node, [param_node])
    ppp = PrePostProcessor(model)
    ppp.input().tensor().set_element_type(ov.Type.u8)
    ppp.input().tensor().set_layout(ov.Layout("NHWC"))
    ppp.input().tensor().set_shape([1, -1, -1, 3])
    ppp.input().preprocess().custom(
        resize_image_with_aspect(
            (8, 8),
            "linear",
            0,
        )
    )
    ppp.input().preprocess().convert_element_type(ov.Type.f32)
    ov_resize_image_with_aspect = ov.Core().compile_model(ppp.build(), "CPU")

    img = np.ones((2, 4, 3), dtype=np.uint8)
    ov_results = ov_resize_image_with_aspect(img[None])
    np_results = resize_image_with_aspect_ocv(img, (8, 8))

    assert np.sum(np.abs(list(ov_results.values())[0][0] - np_results)) < 1e-05
