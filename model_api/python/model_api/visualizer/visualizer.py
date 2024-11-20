"""Visualizer."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from model_api.visualizer.primitives import Label

if TYPE_CHECKING:
    from PIL import Image

    from model_api.visualizer.visualize_mixin import VisualizeMixin

    from .layout import Layout


class Visualizer:
    def __init__(self, layout: Layout | None = None) -> None:
        self.layout = layout

    def show(
        self,
        image: Image,
        result: VisualizeMixin,
    ) -> None:
        result: Image = self._generate(image, result)
        result.show()

    def save(
        self,
        image: Image,
        result: VisualizeMixin,
        path: str,
    ) -> None:
        result: Image = self._generate(image, result)
        result.save(path)

    def _generate(self, image: Image, result: VisualizeMixin) -> Image:
        if self.layout is not None:
            return self.layout(image, result)
        return result.default_layout(image, result)
