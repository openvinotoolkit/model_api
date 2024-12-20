"""Classification Scene."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from PIL import Image

from model_api.models.result.types import ClassificationResult
from model_api.visualizer.layout import Flatten, Layout
from model_api.visualizer.primitive import Overlay
from model_api.visualizer.scene import Scene


class ClassificationScene(Scene):
    """Classification Scene."""

    def __init__(self, image: Image, result: ClassificationResult) -> None:
        self.image = image
        self.result = result

    @property
    def default_layout(self) -> Layout:
        return Flatten(Overlay)
