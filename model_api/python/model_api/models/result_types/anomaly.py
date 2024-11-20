"""Anomaly result type."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import cv2
import numpy as np

from model_api.visualizer.layout import Flatten, Layout
from model_api.visualizer.primitives import BoundingBoxes, Label, Overlay, Polygon

from .base import Result


class AnomalyResult(Result):
    """Results for anomaly models."""

    def __init__(
        self,
        anomaly_map: np.ndarray | None = None,
        pred_boxes: np.ndarray | None = None,
        pred_label: str | None = None,
        pred_mask: np.ndarray | None = None,
        pred_score: float | None = None,
    ) -> None:
        super().__init__()
        self.anomaly_map = anomaly_map
        self.pred_boxes = pred_boxes
        self.pred_label = pred_label
        self.pred_mask = pred_mask
        self.pred_score = pred_score

    def _compute_min_max(self, tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Computes min and max values of the tensor."""
        return tensor.min(), tensor.max()

    def __str__(self) -> str:
        assert self.anomaly_map is not None
        assert self.pred_mask is not None
        anomaly_map_min, anomaly_map_max = self._compute_min_max(self.anomaly_map)
        pred_mask_min, pred_mask_max = self._compute_min_max(self.pred_mask)
        return (
            f"anomaly_map min:{anomaly_map_min} max:{anomaly_map_max};"
            f"pred_score:{np.round(self.pred_score, 1) if self.pred_score else 0.0};"
            f"pred_label:{self.pred_label};"
            f"pred_mask min:{pred_mask_min} max:{pred_mask_max};"
        )

    def _register_primitives(self) -> None:
        """Converts the result to primitives."""
        anomaly_map = cv2.applyColorMap(self.anomaly_map, cv2.COLORMAP_JET)
        self._add_primitive(Overlay(anomaly_map))
        for box in self.pred_boxes:
            self._add_primitive(BoundingBoxes(*box))
        if self.pred_label is not None:
            self._add_primitive(Label(self.pred_label, bg_color="red" if self.pred_label == "Anomaly" else "green"))
        self._add_primitive(Label(f"Score: {self.pred_score}"))
        self._add_primitive(Polygon(mask=self.pred_mask))

    @property
    def default_layout(self) -> Layout:
        return Flatten(
            Overlay,
            Polygon,
            Label,
        )
