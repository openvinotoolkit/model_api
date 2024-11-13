"""Mixin for visualization."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC


class VisualizeMixin(ABC):
    """Mixin for visualization."""

    def get_labels(self):
        """Get labels."""
