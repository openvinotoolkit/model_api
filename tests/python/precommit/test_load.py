#
# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from model_api.models import Model


def test_model_with_unnamed_output_load():
    # the model's output doesn't have a name
    _ = Model.create_model(
        "data/otx_models/tinynet_imagenet.xml",
        model_type="Classification",
        preload=True,
    )
