"""Anomaly result media."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import cv2

from model_api.models.result.types import AnomalyResult
from model_api.visualizer.layout import Flatten, Layout
from model_api.visualizer.media import Media
from model_api.visualizer.primitives import Overlay


class AnomalyMedia(Media):
    """Anomaly result media."""

    def __init__(self, result: AnomalyResult) -> None:
        anomaly_map = cv2.applyColorMap(result.anomaly_map, cv2.COLORMAP_JET)
        super().__init__(Overlay(anomaly_map))

    @property
    def default_layout(self) -> Layout:
        return Flatten(Overlay)
