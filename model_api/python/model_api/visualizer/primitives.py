"""Base class for primitives."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class Primitive(ABC):
    """Primitive class."""

    @abstractmethod
    def compute(self, **kwargs) -> Image:
        pass


class Label(Primitive):
    """Label primitive."""

    def __init__(
        self,
        label: str,
        fg_color: str | tuple[int, int, int] = "black",
        bg_color: str | tuple[int, int, int] = "yellow",
        font_path: str | None | BytesIO = None,
        size: int = 16,
    ) -> None:
        self.label = label
        self.fg_color = fg_color
        self.bg_color = bg_color
        self.font = ImageFont.load_default(size=size) if font_path is None else ImageFont.truetype(font_path, size)

    def compute(self, image: Image, overlay_on_image: bool = True, buffer_y: int = 5) -> Image:
        """Generate label image.

        If overlay_on_image is True, the label will be drawn on top of the image.
        Else only the label will be drawn. This is useful for collecting labels so that they can be drawn on the same
        image.
        """
        dummy_image = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(dummy_image)
        textbox = draw.textbbox((0, 0), self.label, font=self.font)
        label_image = Image.new("RGB", (textbox[2] - textbox[0], textbox[3] + buffer_y - textbox[1]), self.bg_color)
        draw = ImageDraw.Draw(label_image)
        draw.text((0, 0), self.label, font=self.font, fill=self.fg_color)
        if overlay_on_image:
            image.paste(label_image, (0, 0))
            return image
        return label_image

    @classmethod
    def overlay_labels(cls, image: Image, label_images: list[Image], buffer: int = 5) -> Image:
        """Overlay multiple label images on top of the image.

        Paste the labels in a row but wrap the labels if they exceed the image width.
        """
        offset_x = 0
        offset_y = 0
        for label_image in label_images:
            image.paste(label_image, (offset_x, offset_y))
            offset_x += label_image.width + buffer
            if offset_x + label_image.width > image.width:
                offset_x = 0
                offset_y += label_image.height
        return image


class Polygon(Primitive):
    """Polygon primitive."""

    def __init__(
        self,
        points: list[tuple[int, int]] | None = None,
        mask: np.ndarray | None = None,
        color: str | tuple[int, int, int] = "blue",
    ) -> None:
        self.points = self._get_points(points, mask)
        self.color = color

    def _get_points(self, points: list[tuple[int, int]] | None, mask: np.ndarray | None) -> list[tuple[int, int]]:
        if points is not None:
            return points
        return self._get_points_from_mask(mask)

    def _get_points_from_mask(self, mask: np.ndarray) -> list[tuple[int, int]]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        _points = contours[0].squeeze().tolist()
        return [tuple(point) for point in _points]

    def compute(self, image: Image) -> Image:
        draw = ImageDraw.Draw(image)
        draw.polygon(self.points, fill=self.color)
        return image


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


class BoundingBoxes(Primitive):
    def __init__(self, x1: int, y1: int, x2: int, y2: int, color: str | tuple[int, int, int] = "blue") -> None:
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.color = color

    def compute(self, image: Image) -> Image:
        draw = ImageDraw.Draw(image)
        draw.rectangle([self.x1, self.y1, self.x2, self.y2], fill=None, outline=self.color, width=2)
        return image
