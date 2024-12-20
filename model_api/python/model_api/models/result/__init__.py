"""Model results."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .scene import (
    AnomalyScene,
    ClassificationScene,
    DetectionScene,
    KeypointScene,
    SegmentationScene,
    VisualPromptingScene,
)
from .types import (
    AnomalyResult,
    ClassificationResult,
    Contour,
    DetectedKeypoints,
    DetectionResult,
    ImageResultWithSoftPrediction,
    InstanceSegmentationResult,
    Label,
    PredictedMask,
    Result,
    RotatedSegmentationResult,
    VisualPromptingResult,
    ZSLVisualPromptingResult,
)

__all__ = [
    "AnomalyResult",
    "ClassificationResult",
    "Contour",
    "DetectionResult",
    "DetectedKeypoints",
    "ImageResultWithSoftPrediction",
    "InstanceSegmentationResult",
    "Label",
    "PredictedMask",
    "Result",
    "VisualPromptingResult",
    "ZSLVisualPromptingResult",
    "RotatedSegmentationResult",
    "AnomalyScene",
    "ClassificationScene",
    "DetectionScene",
    "KeypointScene",
    "SegmentationScene",
    "VisualPromptingScene",
]
