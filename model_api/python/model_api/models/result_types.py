"""Result types."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC
from typing import NamedTuple

import cv2
import numpy as np

from model_api.visualizer.primitives import BoundingBoxes, Label, Overlay, Polygon
from model_api.visualizer.visualize_mixin import VisualizeMixin


class Result(VisualizeMixin, ABC):
    """Base result type."""


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
        self._add_primitive(Label(self.pred_label, bg_color="red" if self.pred_label == "Anomaly" else "green"))
        self._add_primitive(Label(f"Score: {self.pred_score}"))
        self._add_primitive(Polygon(mask=self.pred_mask))


class ClassificationResult(Result):
    """Results for classification models."""

    def __init__(
        self,
        top_labels: list[tuple[int, str, float]] | None = None,
        saliency_map: np.ndarray | None = None,
        feature_vector: np.ndarray | None = None,
        raw_scores: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.top_labels = top_labels
        self.saliency_map = saliency_map
        self.feature_vector = feature_vector
        self.raw_scores = raw_scores

    def __str__(self) -> str:
        assert self.top_labels is not None
        labels = ", ".join(f"{idx} ({label}): {confidence:.3f}" for idx, label, confidence in self.top_labels)
        return (
            f"{labels}, {_array_shape_to_str(self.saliency_map)}, {_array_shape_to_str(self.feature_vector)}, "
            f"{_array_shape_to_str(self.raw_scores)}"
        )

    def _register_primitives(self) -> None:
        # TODO add saliency map
        for idx, label, confidence in self.top_labels:
            self._add_primitive(Label(f"Rank: {idx}, {label}: {confidence:.3f}"))


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
        return f"{obj_str}{_array_shape_to_str(self.saliency_map)}; {_array_shape_to_str(self.feature_vector)}"


class DetectedKeypoints:
    def __init__(self, keypoints: np.ndarray, scores: np.ndarray) -> None:
        self.keypoints = keypoints
        self.scores = scores

    def __str__(self):
        return (
            f"keypoints: {self.keypoints.shape}, "
            f"keypoints_x_sum: {np.sum(self.keypoints[:, :1]):.3f}, "
            f"scores: {self.scores.shape}"
        )


class SegmentedObject:
    def __init__(
        self,
        xmin: int,
        ymin: int,
        xmax: int,
        ymax: int,
        score: float,
        id: int,
        mask: np.ndarray,
        str_label: str | None = None,
    ) -> None:
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.score = score
        self.id = id
        self.str_label = str_label
        self.mask = mask

    def __str__(self):
        return f"{super().__str__()}, {(self.mask > 0.5).sum()}"


class SegmentedObjectWithRects(SegmentedObject):
    def __init__(self, segmented_object: SegmentedObject, rotated_rect: RotatedRect) -> None:
        super().__init__(
            segmented_object.xmin,
            segmented_object.ymin,
            segmented_object.xmax,
            segmented_object.ymax,
            segmented_object.score,
            segmented_object.id,
            segmented_object.str_label,
            segmented_object.mask,
        )
        self.rotated_rect = rotated_rect

    def __str__(self):
        res = super().__str__()
        rect = self.rotated_rect
        res += f", RotatedRect: {rect[0][0]:.3f} {rect[0][1]:.3f} {rect[1][0]:.3f} {rect[1][1]:.3f} {rect[2]:.3f}"
        return res


class InstanceSegmentationResult:
    def __init__(
        self,
        segmentedObjects: list[SegmentedObject | SegmentedObjectWithRects],
        saliency_map: list[np.ndarray],
        feature_vector: np.ndarray,
    ) -> None:
        self.segmentedObjects = segmentedObjects
        # Contain per class saliency_maps and "feature_vector" model output if feature_vector exists
        self.saliency_map = saliency_map
        self.feature_vector = feature_vector

    def __str__(self):
        obj_str = "; ".join(str(obj) for obj in self.segmentedObjects)
        filled = 0
        for cls_map in self.saliency_map:
            if cls_map.size:
                filled += 1
        prefix = f"{obj_str}; " if len(obj_str) else ""
        return prefix + f"{filled}; [{','.join(str(i) for i in self.feature_vector.shape)}]"


class Contour:
    def __init__(
        self,
        label: str,
        probability: float,
        shape: list[tuple[int, int]],
    ) -> None:
        self.label = label
        self.probability = probability
        self.shape = shape

    def __str__(self):
        return f"{self.label}: {self.probability:.3f}, {len(self.shape)}"


class ImageResultWithSoftPrediction:
    def __init__(
        self,
        resultImage: np.ndarray,
        soft_prediction: np.ndarray,
        saliency_map: np.ndarray,
        feature_vector: np.ndarray,
    ) -> None:
        self.resultImage = resultImage
        self.soft_prediction = soft_prediction
        # Contain per class saliency_maps and "feature_vector" model output if feature_vector exists
        self.saliency_map = saliency_map  # Requires return_soft_prediction==True
        self.feature_vector = feature_vector

    def __str__(self):
        outHist = cv2.calcHist(
            [self.resultImage.astype(np.uint8)],
            channels=None,
            mask=None,
            histSize=[256],
            ranges=[0, 255],
        )
        hist = ""
        for i, count in enumerate(outHist):
            if count > 0:
                hist += f"{i}: {count[0] / self.resultImage.size:.3f}, "
        return (
            f"{hist}{_array_shape_to_str(self.soft_prediction)}, "
            f"{_array_shape_to_str(self.saliency_map)}, "
            f"{_array_shape_to_str(self.feature_vector)}"
        )


class VisualPromptingResult:
    def __init__(
        self,
        upscaled_masks: list[np.ndarray] | None = None,
        processed_mask: list[np.ndarray] | None = None,
        low_res_masks: list[np.ndarray] | None = None,
        iou_predictions: list[np.ndarray] | None = None,
        scores: list[np.ndarray] | None = None,
        labels: list[np.ndarray] | None = None,
        hard_predictions: list[np.ndarray] | None = None,
        soft_predictions: list[np.ndarray] | None = None,
        best_iou: list[float] | None = None,
    ) -> None:
        self.upscaled_masks = upscaled_masks
        self.processed_mask = processed_mask
        self.low_res_masks = low_res_masks
        self.iou_predictions = iou_predictions
        self.scores = scores
        self.labels = labels
        self.hard_predictions = hard_predictions
        self.soft_predictions = soft_predictions
        self.best_iou = best_iou

    def _compute_min_max(self, tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return tensor.min(), tensor.max()

    def __str__(self) -> str:
        assert self.hard_predictions is not None
        assert self.upscaled_masks is not None
        upscaled_masks_min, upscaled_masks_max = self._compute_min_max(
            self.upscaled_masks[0],
        )

        return (
            f"upscaled_masks min:{upscaled_masks_min:.3f} max:{upscaled_masks_max:.3f};"
            f"hard_predictions shape:{self.hard_predictions[0].shape};"
        )


class PredictedMask:
    def __init__(
        self,
        mask: list[np.ndarray],
        points: list[np.ndarray] | np.ndarray,
        scores: list[float] | np.ndarray,
    ) -> None:
        self.mask = mask
        self.points = points
        self.scores = scores

    def __str__(self) -> str:
        obj_str = ""
        obj_str += f"mask sum: {np.sum(sum(self.mask))}; "

        if isinstance(self.points, list):
            for i, point in enumerate(self.points):
                obj_str += "["
                obj_str += ", ".join(str(round(c, 2)) for c in point)
                obj_str += "] "
                obj_str += "iou: " + f"{float(self.scores[i]):.3f} "
        else:
            for i in range(self.points.shape[0]):
                point = self.points[i]
                obj_str += "["
                obj_str += ", ".join(str(round(c, 2)) for c in point)
                obj_str += "] "
                obj_str += "iou: " + f"{float(self.scores[i]):.3f} "

        return obj_str.strip()


class ZSLVisualPromptingResult:
    def __init__(self, data: dict[int, PredictedMask]) -> None:
        self.data: dict[int, PredictedMask]

    def __str__(self) -> str:
        return ", ".join(str(self.data[k]) for k in self.data)

    def get_mask(self, label: int) -> PredictedMask:
        """Returns a mask belonging to a given label"""
        return self.data[label]


def _array_shape_to_str(array: np.ndarray | None) -> str:
    if array is not None:
        return f"[{','.join(str(i) for i in array.shape)}]"
    return "[]"
