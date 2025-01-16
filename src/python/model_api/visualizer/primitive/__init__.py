"""Primitive classes."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .bounding_box import BoundingBox
from .overlay import Overlay
from .polygon import Polygon
from .primitive import Primitive

__all__ = ["Primitive", "BoundingBox", "Overlay", "Polygon"]
