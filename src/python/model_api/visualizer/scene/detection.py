"""Detection Scene."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from itertools import starmap
from typing import Union

import cv2
from PIL import Image

from model_api.models.result import DetectionResult
from model_api.visualizer.layout import Flatten, HStack, Layout
from model_api.visualizer.primitive import BoundingBox, Label, Overlay

from .scene import Scene


class DetectionScene(Scene):
    """Detection Scene."""

    def __init__(self, image: Image, result: DetectionResult, layout: Union[Layout, None] = None) -> None:
        super().__init__(
            base=image,
            label=self._get_labels(result),
            bounding_box=self._get_bounding_boxes(result),
            overlay=self._get_overlays(result),
            layout=layout,
        )

    def _get_labels(self, result: DetectionResult) -> list[Label]:
        labels = []
        for label, score, label_name in zip(result.labels, result.scores, result.label_names):
            labels.append(Label(label=f"{label} {label_name}", score=score))
        return labels

    def _get_overlays(self, result: DetectionResult) -> list[Overlay]:
        overlays = []
        for saliency_map in result.saliency_map[0][1:]:  # Assumes only one batch. Skip background class.
            saliency_map = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
            overlays.append(Overlay(saliency_map))
        return overlays

    def _get_bounding_boxes(self, result: DetectionResult) -> list[BoundingBox]:
        return list(starmap(BoundingBox, result.bboxes))

    @property
    def default_layout(self) -> Layout:
        return HStack(Flatten(BoundingBox, Label), Overlay)
