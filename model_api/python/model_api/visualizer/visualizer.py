"""Visualizer."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

    from .layout import Layout
    from .media import Media


class Visualizer:
    def __init__(self, layout: Layout | None = None) -> None:
        self.layout = layout

    def show(
        self,
        image: Image,
        result: Media,
    ) -> None:
        result: Image = self._generate(image, result)
        result.show()

    def save(
        self,
        image: Image,
        result: Media,
        path: str,
    ) -> None:
        result: Image = self._generate(image, result)
        result.save(path)

    def _generate(self, image: Image, result: Media) -> Image:
        if self.layout is not None:
            return self.layout(image, result)
        return result.default_layout(image, result)
