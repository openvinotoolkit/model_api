"""Visualizer."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from PIL import Image

from model_api.visualizer.visualize_mixin import VisualizeMixin


class Visualizer:

    def show(self, image: Image, result: VisualizeMixin) -> None:
        pass
