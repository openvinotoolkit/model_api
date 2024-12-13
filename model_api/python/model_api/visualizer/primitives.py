"""Base class for primitives."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from io import BytesIO
from typing import TYPE_CHECKING

import cv2
from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    import numpy as np


class Primitive(ABC):
    """Primitive class."""

    @abstractmethod
    def compute(self, **kwargs) -> Image:
        pass


class Overlay(Primitive):
    """Overlay an image.

    Useful for XAI and Anomaly Maps.
    """

    def __init__(self, image: Image | np.ndarray, opacity: float = 0.4) -> None:
        self.image = self._to_image(image)
        self.opacity = opacity

    def _to_image(self, image: Image | np.ndarray) -> Image:
        if isinstance(image, Image.Image):
            return image
        return Image.fromarray(image)

    def compute(self, image: Image) -> Image:
        _image = self.image.resize(image.size)
        return Image.blend(image, _image, self.opacity)
