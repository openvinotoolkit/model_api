"""Mixin for visualization."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Type

from .layout import Layout
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

    @property
    @abstractmethod
    def default_layout(self) -> Layout:
        """Default layout."""

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

    def has_primitive(self, primitive: Type[Primitive]) -> bool:
        """Check if the primitive type is registered."""
        self._register_primitives_if_needed()
        if primitive == Label:
            return bool(self._labels)
        if primitive == Polygon:
            return bool(self._polygons)
        if primitive == Overlay:
            return bool(self._overlays)
        if primitive == BoundingBoxes:
            return bool(self._bounding_boxes)
        return False

    def get_primitive(self, primitive: Type[Primitive]) -> Primitive:
        """Get primitive."""
        self._register_primitives_if_needed()
        if primitive == Label:
            return self._labels
        if primitive == Polygon:
            return self._polygons
        if primitive == Overlay:
            return self._overlays
        if primitive == BoundingBoxes:
            return self._bounding_boxes
        msg = f"Primitive {primitive} not found"
        raise ValueError(msg)

    def _register_primitives_if_needed(self):
        if not self._registered_primitives:
            self._register_primitives()
            self._registered_primitives = True
