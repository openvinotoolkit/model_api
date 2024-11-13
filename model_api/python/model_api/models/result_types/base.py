"""Base result type."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from typing import NamedTuple

from model_api.visualizer.visualize_mixin import VisualizeMixin


class Result(VisualizeMixin, ABC):
    """Base result type."""
