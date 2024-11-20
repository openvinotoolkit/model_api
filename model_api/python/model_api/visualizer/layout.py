"""Visualization Layout"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Type

from PIL import Image

if TYPE_CHECKING:
    from model_api.visualizer.primitives import Primitive

    from .visualize_mixin import VisualizeMixin


class Layout(ABC):
    """Base class for layouts."""

    def _compute_on_primitive(self, primitive: Primitive, image: Image, result: VisualizeMixin) -> Image | None:
        if result.has_primitive(primitive):
            primitives = result.get_primitive(primitive)
            for primitive in primitives:
                image = primitive.compute(image)
            return image
        return None


class HStack(Layout):
    """Horizontal stack layout."""

    def __init__(self, *args: Layout | Type[Primitive]) -> None:
        self.children = args

    def __call__(self, image: Image, result: VisualizeMixin) -> Image:
        images: list[Image] = []
        for child in self.children:
            if isinstance(child, Layout):
                images.append(child(image, result))
            else:
                _image = image.copy()
                _image = self._compute_on_primitive(child, _image, result)
                if _image is not None:
                    images.append(_image)
        return self._stitch(*images)

    def _stitch(self, *images: Image) -> Image:
        """Stitch images together.

        Args:
            images (Image): Images to stitch.

        Returns:
            Image: Stitched image.
        """
        new_image = Image.new(
            "RGB",
            (
                sum(image.width for image in images),
                max(image.height for image in images),
            ),
        )
        x_offset = 0
        for image in images:
            new_image.paste(image, (x_offset, 0))
            x_offset += image.width
        return new_image


class VStack(Layout):
    """Vertical stack layout."""


class Flatten(Layout):
    """Put all primitives on top of each other"""

    def __init__(self, *args: Type[Primitive]) -> None:
        self.children = args

    def __call__(self, image: Image, result: VisualizeMixin) -> Image:
        _image: Image = image.copy()
        for child in self.children:
            _image = self._compute_on_primitive(child, _image, result)
        return _image
