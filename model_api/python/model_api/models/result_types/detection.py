"""Detection result type."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from .utils import array_shape_to_str


def find_layer_by_name(name, layers):
    suitable_layers = [layer_name for layer_name in layers if name in layer_name]
    if not suitable_layers:
        msg = f'Suitable layer for "{name}" output is not found'
        raise ValueError(msg)

    if len(suitable_layers) > 1:
        msg = f'More than 1 layer matched to "{name}" output'
        raise ValueError(msg)

    return suitable_layers[0]


class DetectionResult:
    """Result for detection model.

    Args:
        bboxes (np.ndarray): bounding boxes in dim (N, 4) where N is the number of boxes.
        labels (np.ndarray): labels for each bounding box in dim (N,).
        scores (np.ndarray| None, optional): confidence scores for each bounding box in dim (N,).
        label_names (list[str] | None, optional): class names for each label. Defaults to None.
        saliency_map (np.ndarray | None, optional): saliency map for XAI. Defaults to None.
        feature_vector (np.ndarray | None, optional): feature vector for XAI. Defaults to None.
    """

    def __init__(
        self,
        bboxes: np.ndarray,
        labels: np.ndarray,
        scores: np.ndarray | None = None,
        label_names: list[str] | None = None,
        saliency_map: np.ndarray | None = None,
        feature_vector: np.ndarray | None = None,
    ):
        super().__init__()
        self._bboxes = bboxes
        self._labels = labels
        self._scores = scores if scores is not None else np.zeros(len(bboxes))
        self._label_names = ["#"] * len(bboxes) if label_names is None else label_names
        self._saliency_map = saliency_map
        self._feature_vector = feature_vector

    def __len__(self) -> int:
        return len(self._bboxes)

    def __str__(self) -> str:
        return (
            f"Num of boxes: {self._bboxes.shape}, "
            f"Num of labels: {len(self._labels)}, "
            f"Num of scores: {len(self._scores)}, "
            f"Saliency Map: {array_shape_to_str(self._saliency_map)}, "
            f"Feature Vec: {array_shape_to_str(self._feature_vector)}"
        )

    def get_obj_sizes(self) -> np.ndarray:
        """Get object sizes.

        Returns:
            np.ndarray: Object sizes in dim of (N,).
        """
        return (self._bboxes[:, 2] - self._bboxes[:, 0]) * (self._bboxes[:, 3] - self._bboxes[:, 1])

    @property
    def bboxes(self) -> np.ndarray:
        return self._bboxes

    @bboxes.setter
    def bboxes(self, value):
        if not isinstance(value, np.ndarray):
            msg = "Bounding boxes must be numpy array."
            raise ValueError(msg)
        self._bboxes = value

    @property
    def labels(self) -> np.ndarray:
        return self._labels

    @labels.setter
    def labels(self, value):
        if not isinstance(value, np.ndarray):
            msg = "Labels must be numpy array."
            raise ValueError(msg)
        self._labels = value

    @property
    def scores(self) -> np.ndarray:
        return self._scores

    @scores.setter
    def scores(self, value):
        if not isinstance(value, np.ndarray):
            msg = "Scores must be numpy array."
            raise ValueError(msg)
        self._scores = value

    @property
    def label_names(self) -> list[str]:
        return self._label_names

    @label_names.setter
    def label_names(self, value):
        if not isinstance(value, list):
            msg = "Label names must be list."
            raise ValueError(msg)
        self._label_names = value

    @property
    def saliency_map(self) -> np.ndarray:
        return self._saliency_map

    @saliency_map.setter
    def saliency_map(self, value):
        if not isinstance(value, np.ndarray):
            msg = "Saliency map must be numpy array."
            raise ValueError(msg)
        self._saliency_map = value

    @property
    def feature_vector(self) -> np.ndarray:
        return self._feature_vector

    @feature_vector.setter
    def feature_vector(self, value):
        if not isinstance(value, np.ndarray):
            msg = "Feature vector must be numpy array."
            raise ValueError(msg)
        self._feature_vector = value


class SingleOutputParser:
    def __init__(self, all_outputs):
        if len(all_outputs) != 1:
            msg = "Network must have only one output."
            raise ValueError(msg)
        self.output_name, output_data = next(iter(all_outputs.items()))
        last_dim = output_data.shape[-1]
        if last_dim != 7:
            msg = f"The last dimension of the output blob must be equal to 7, got {last_dim} instead."
            raise ValueError(msg)

    def __call__(self, outputs) -> DetectionResult:
        """Parse model outputs.

        Args:
            outputs (dict): Model outputs wrapped in dict.

        Returns:
            DetectionResult: Parsed model outputs.
        """
        bboxes = []
        scores = []
        labels = []
        for _, label, score, xmin, ymin, xmax, ymax in outputs[self.output_name][0][0]:
            bboxes.append((xmin, ymin, xmax, ymax))
            scores.append(score)
            labels.append(label)
        bboxes = np.array(bboxes)
        scores = np.array(scores)
        labels = np.array(labels)
        return DetectionResult(bboxes, scores, labels)


class MultipleOutputParser:
    def __init__(
        self,
        layers,
        bboxes_layer="bboxes",
        scores_layer="scores",
        labels_layer="labels",
    ):
        self.labels_layer = find_layer_by_name(labels_layer, layers)
        self.scores_layer = find_layer_by_name(scores_layer, layers)
        self.bboxes_layer = find_layer_by_name(bboxes_layer, layers)

    def __call__(self, outputs) -> DetectionResult:
        """Parse model outputs.

        Args:
            outputs (dict): Model outputs wrapped in dict.

        Returns:
            DetectionResult: Parsed model outputs.
        """
        bboxes = np.array(outputs[self.bboxes_layer][0])
        scores = np.array(outputs[self.scores_layer][0])
        labels = np.array(outputs[self.labels_layer][0])
        return DetectionResult(bboxes, scores, labels)


class BoxesLabelsParser:
    def __init__(self, layers, input_size, labels_layer="labels", default_label=0):
        try:
            self.labels_layer = find_layer_by_name(labels_layer, layers)
        except ValueError:
            self.labels_layer = None
            self.default_label = default_label

        self.bboxes_layer = self.find_layer_bboxes_output(layers)
        self.input_size = input_size

    @staticmethod
    def find_layer_bboxes_output(layers):
        filter_outputs = [
            name
            for name, data in layers.items()
            if (len(data.shape) == 2 or len(data.shape) == 3) and data.shape[-1] == 5
        ]
        if not filter_outputs:
            msg = "Suitable output with bounding boxes is not found"
            raise ValueError(msg)
        if len(filter_outputs) > 1:
            msg = "More than 1 candidate for output with bounding boxes."
            raise ValueError(msg)
        return filter_outputs[0]

    def __call__(self, outputs) -> DetectionResult:
        """Parse model outputs.

        Note: Bounding boxes layer from outputs are expected to be in format [xmin, ymin, xmax, ymax, score].

        Args:
            outputs (dict): Model outputs wrapped in dict.

        Returns:
            DetectionResult: Parsed model outputs.
        """
        bboxes = outputs[self.bboxes_layer]
        bboxes = bboxes.squeeze(0)
        scores = bboxes[:, 4]
        bboxes = bboxes[:, :4]
        bboxes[:, 0::2] /= self.input_size[0]
        bboxes[:, 1::2] /= self.input_size[1]
        if self.labels_layer:
            labels = outputs[self.labels_layer]
        else:
            labels = np.full(len(bboxes), self.default_label, dtype=bboxes.dtype)
        labels = labels.squeeze(0)

        return DetectionResult(
            bboxes=bboxes,
            labels=labels,
            scores=scores,
        )
