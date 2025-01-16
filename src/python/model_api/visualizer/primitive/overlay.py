"""Overlay primitive."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import PIL

from .primitive import Primitive


class Overlay(Primitive):
    """Overlay primitive.

    Useful for XAI and Anomaly Maps.

    Args:
        image (PIL.Image | np.ndarray): Image to be overlaid.
        opacity (float): Opacity of the overlay.
    """

    def __init__(self, image: PIL.Image | np.ndarray, opacity: float = 0.4) -> None:
        self.image = self._to_pil(image)
        self.opacity = opacity

    def _to_pil(self, image: PIL.Image | np.ndarray) -> PIL.Image:
        if isinstance(image, np.ndarray):
            return PIL.Image.fromarray(image)
        return image

    def compute(self, image: PIL.Image) -> PIL.Image:
        image_ = self.image.resize(image.size)
        return PIL.Image.blend(image, image_, self.opacity)
