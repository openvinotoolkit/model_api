"""Visualizer."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum

from PIL import Image

from model_api.visualizer.primitives import Label
from model_api.visualizer.visualize_mixin import VisualizeMixin


class VisualizationType(Enum):
    """Visualization type."""

    FULL = "full"
    SIMPLE = "simple"


class Visualizer:
    def __init__(self) -> None:
        # TODO: add transforms for the source image so that it has the same crop, and size as the model.
        pass

    def show(
        self,
        image: Image,
        result: VisualizeMixin,
        visualization_type: VisualizationType | str = VisualizationType.SIMPLE,
    ) -> None:
        visualization_type = VisualizationType(visualization_type)
        result: Image = self._generate(image, result, visualization_type)
        result.show()

    def save(
        self,
        image: Image,
        result: VisualizeMixin,
        path: str,
        visualization_type: VisualizationType | str = VisualizationType.SIMPLE,
    ) -> None:
        visualization_type = VisualizationType(visualization_type)
        result: Image = self._generate(image, result, visualization_type)
        result.save(path)

    def _generate(self, image: Image, result: VisualizeMixin, visualization_type: VisualizationType) -> Image:
        result: Image
        if visualization_type == VisualizationType.SIMPLE:
            result = self._generate_simple(image, result)
        else:
            result = self._generate_full(image, result)
        return result

    def _generate_simple(self, image: Image, result: VisualizeMixin) -> Image:
        """Return a single image with stacked visualizations."""
        # 1. Use Overlay
        _image = image.copy()
        if result.has_overlays:
            overlays = result.get_overlays()
            for overlay in overlays:
                image = overlay.compute(_image)

        elif result.has_polygons:  # 2. else use polygons
            polygons = result.get_polygons()
            for polygon in polygons:
                image = polygon.compute(_image)

        elif result.has_bounding_boxes:  # 3. else use bounding boxes
            bounding_boxes = result.get_bounding_boxes()
            for bounding_box in bounding_boxes:
                image = bounding_box.compute(_image)

        # Finally add labels
        if result.has_labels:
            labels = result.get_labels()
            label_images = []
            for label in labels:
                label_images.append(label.compute(_image, overlay_on_image=False))
            _image = Label.overlay_labels(_image, label_images)

        return _image

    def _generate_full(self, image: Image, result: VisualizeMixin) -> Image:
        """Return a single image with visualizations side by side."""
        images: list[Image] = [image]

        if result.has_overlays:
            overlays = result.get_overlays()
            _image = image.copy()
            for overlay in overlays:
                _image = overlay.compute(_image)
            images.append(_image)
        if result.has_polygons:
            polygons = result.get_polygons()
            _image = image.copy()
            for polygon in polygons:
                _image = polygon.compute(_image)
            images.append(_image)
        if result.has_bounding_boxes:
            bounding_boxes = result.get_bounding_boxes()
            _image = image.copy()
            for bounding_box in bounding_boxes:
                _image = bounding_box.compute(_image)
            images.append(_image)
        if result.has_labels:
            labels = result.get_labels()
            for label in labels:
                images.append(label.compute(image.copy(), overlay_on_image=True))
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
