"""Polygon primitive."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
from PIL import Image, ImageDraw

from .primitive import Primitive

if TYPE_CHECKING:
    import numpy as np


class Polygon(Primitive):
    """Polygon primitive.

    Args:
        points: List of points.
        mask: Mask to draw the polygon.
        color: Color of the polygon.

    Examples:
        >>> polygon = Polygon(points=[(10, 10), (100, 10), (100, 100), (10, 100)], color="red")
        >>> polygon = Polygon(mask=mask, color="red")
        >>> polygon.compute(image).save("polygon.jpg")

        >>> polygon = Polygon(mask=mask, color="red")
        >>> polygon.compute(image).save("polygon.jpg")
    """

    def __init__(
        self,
        points: list[tuple[int, int]] | None = None,
        mask: np.ndarray | None = None,
        color: str | tuple[int, int, int] = "blue",
    ) -> None:
        self.points = self._get_points(points, mask)
        self.color = color

    def _get_points(self, points: list[tuple[int, int]] | None, mask: np.ndarray | None) -> list[tuple[int, int]]:
        """Get points from either points or mask.
        Note:
            Either points or mask should be provided.

        Args:
            points: List of points.
            mask: Mask to draw the polygon.

        Returns:
            List of points.
        """
        if points is not None and mask is not None:
            msg = "Either points or mask should be provided, not both."
            raise ValueError(msg)
        if points is not None:
            points_ = points
        elif mask is not None:
            points_ = self._get_points_from_mask(mask)
        else:
            msg = "Either points or mask should be provided."
            raise ValueError(msg)
        return points_

    def _get_points_from_mask(self, mask: np.ndarray) -> list[tuple[int, int]]:
        """Get points from mask.

        Args:
            mask: Mask to draw the polygon.

        Returns:
            List of points.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        points_ = contours[0].squeeze().tolist()
        return [tuple(point) for point in points_]

    def compute(self, image: Image) -> Image:
        """Compute the polygon.

        Args:
            image: Image to draw the polygon on.

        Returns:
            Image with the polygon drawn on it.
        """
        draw = ImageDraw.Draw(image)
        draw.polygon(self.points, fill=self.color)
        return image
