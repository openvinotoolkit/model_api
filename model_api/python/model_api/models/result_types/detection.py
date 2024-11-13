"""Detection result type."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import numpy as np

from model_api.models.result_types.base import Result

from .utils import array_shape_to_str


class Detection:
    def __init__(self, xmin, ymin, xmax, ymax, score, id, str_label=None) -> None:
        self.xmin: int = xmin
        self.ymin: int = ymin
        self.xmax: int = xmax
        self.ymax: int = ymax
        self.score: float = score
        self.id: int = int(id)
        self.str_label: str | None = str_label

    def __str__(self):
        return f"{self.xmin}, {self.ymin}, {self.xmax}, {self.ymax}, {self.id} ({self.str_label}): {self.score:.3f}"


class DetectionResult(Detection, Result):
    """Result for detection model."""

    objects: list[Detection] | None = None
    saliency_map: np.ndarray | None = None
    feature_vector: np.ndarray | None = None

    def __str__(self):
        assert self.objects is not None
        obj_str = "; ".join(str(obj) for obj in self.objects)
        if obj_str:
            obj_str += "; "
        return f"{obj_str}{array_shape_to_str(self.saliency_map)}; {array_shape_to_str(self.feature_vector)}"
