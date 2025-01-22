"""Detection Scene."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Union

from PIL import Image

from model_api.models.result import DetectionResult
from model_api.visualizer.layout import Layout

from .scene import Scene


class DetectionScene(Scene):
    """Detection Scene."""

    def __init__(self, image: Image, result: DetectionResult, layout: Union[Layout, None] = None) -> None:
        self.image = image
        self.result = result
