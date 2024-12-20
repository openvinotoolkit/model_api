"""Keypoint Scene."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from model_api.models.result.types import DetectedKeypoints
from model_api.visualizer.layout import Flatten, Layout
from model_api.visualizer.primitive import Overlay
from model_api.visualizer.scene import Scene


class KeypointScene(Scene):
    """Keypoint Scene."""

    def __init__(self, result: DetectedKeypoints) -> None:
        self.result = result

    @property
    def default_layout(self) -> Layout:
        return Flatten(Overlay)
