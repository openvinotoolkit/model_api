"""Visualization Layout"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from PIL import Image

    from model_api.visualizer.primitives import Primitive

    from .media import Media


class Layout(ABC):
    """Base class for layouts."""

    def _compute_on_primitive(self, primitive: Primitive, image: Image, media: Media) -> Image | None:
        if media.has_primitive(primitive):
            primitives = media.get_primitive(primitive)
            for primitive in primitives:
                image = primitive.compute(image)
            return image
        return None


class Flatten(Layout):
    """Put all primitives on top of each other"""

    def __init__(self, *args: Type[Primitive]) -> None:
        self.children = args

    def __call__(self, image: Image, media: Media) -> Image:
        _image: Image = image.copy()
        for child in self.children:
            _image = self._compute_on_primitive(child, _image, media)
        return _image
