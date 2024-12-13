"""Media object."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from typing import Type

from .layout import Layout
from .primitives import Overlay, Primitive


class Media:
    """Media object.

    Media object that is used by the visualizer to render the prediction.

    Args:
        *args: Primitives to be added to the prediction.

    Example:
        >>> media = Media(Label("Label"), BoundingBoxes(0, 0, 10, 10))
    """

    def __init__(self, *args: Primitive) -> None:
        self._overlays: list[Overlay] = []
        self._add_primitives(args)

    def _add_primitives(self, primitives: list[Primitive]) -> None:
        """Add primitives to the prediction."""
        for primitive in primitives:
            self._add_primitive(primitive)

    def _add_primitive(self, primitive: Primitive) -> None:
        """Add primitive."""
        if isinstance(primitive, Overlay):
            self._overlays.append(primitive)
        else:
            msg = f"Primitive {primitive} not supported"
            raise ValueError(msg)

    def has_primitive(self, primitive: Type[Primitive]) -> bool:
        """Check if the primitive type is registered."""
        if primitive == Overlay:
            return bool(self._overlays)
        return False

    def get_primitive(self, primitive: Type[Primitive]) -> Primitive:
        """Get primitive."""
        if primitive == Overlay:
            return self._overlays
        msg = f"Primitive {primitive} not found"
        raise ValueError(msg)

    @property
    @abstractmethod
    def default_layout(self) -> Layout:
        """Default layout for the media."""
