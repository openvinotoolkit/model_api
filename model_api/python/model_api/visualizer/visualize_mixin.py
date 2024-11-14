"""Mixin for visualization."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

from .primitives import BoundingBoxes, Label, Overlay, Polygon, Primitive


class VisualizeMixin(ABC):
    """Mixin for visualization."""

    def __init__(self) -> None:
        self._labels = []
        self._polygons = []
        self._overlays = []
        self._bounding_boxes = []
        self._registered_primitives = False

    @abstractmethod
    def _register_primitives(self) -> None:
        """Convert result entities to primitives."""

    def _add_primitive(self, primitive: Primitive) -> None:
        """Add primitive."""
        if isinstance(primitive, Label):
            self._labels.append(primitive)
        elif isinstance(primitive, Polygon):
            self._polygons.append(primitive)
        elif isinstance(primitive, Overlay):
            self._overlays.append(primitive)
        elif isinstance(primitive, BoundingBoxes):
            self._bounding_boxes.append(primitive)

    @property
    def has_labels(self) -> bool:
        """Check if there are labels."""
        self._register_primitives_if_needed()
        return bool(self._labels)

    @property
    def has_bounding_boxes(self) -> bool:
        """Check if there are bounding boxes."""
        self._register_primitives_if_needed()
        return bool(self._bounding_boxes)

    @property
    def has_polygons(self) -> bool:
        """Check if there are polygons."""
        self._register_primitives_if_needed()
        return bool(self._polygons)

    @property
    def has_overlays(self) -> bool:
        """Check if there are overlays."""
        self._register_primitives_if_needed()
        return bool(self._overlays)

    def get_labels(self) -> list[Label]:
        """Get labels."""
        self._register_primitives_if_needed()
        return self._labels

    def get_polygons(self) -> list[Polygon]:
        """Get polygons."""
        self._register_primitives_if_needed()
        return self._polygons

    def get_overlays(self) -> list[Overlay]:
        """Get overlays."""
        self._register_primitives_if_needed()
        return self._overlays

    def get_bounding_boxes(self) -> list[BoundingBoxes]:
        """Get bounding boxes."""
        self._register_primitives_if_needed()
        return self._bounding_boxes

    def _register_primitives_if_needed(self):
        if not self._registered_primitives:
            self._register_primitives()
            self._registered_primitives = True
