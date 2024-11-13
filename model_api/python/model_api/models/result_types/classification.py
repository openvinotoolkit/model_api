"""Classification result type."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from model_api.models.result_types.base import Result

from .utils import array_shape_to_str


class ClassificationResult(Result):
    """Results for classification models."""

    def __init__(
        self,
        top_labels: list[tuple[int, str, float]] | None = None,
        saliency_map: np.ndarray | None = None,
        feature_vector: np.ndarray | None = None,
        raw_scores: np.ndarray | None = None,
    ) -> None:
        self.top_labels = top_labels
        self.saliency_map = saliency_map
        self.feature_vector = feature_vector
        self.raw_scores = raw_scores

    def __str__(self) -> str:
        assert self.top_labels is not None
        labels = ", ".join(f"{idx} ({label}): {confidence:.3f}" for idx, label, confidence in self.top_labels)
        return (
            f"{labels}, {array_shape_to_str(self.saliency_map)}, {array_shape_to_str(self.feature_vector)}, "
            f"{array_shape_to_str(self.raw_scores)}"
        )
