"""Model results."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .types import (
    AnomalyResult,
    ClassificationResult,
    Contour,
    DetectedKeypoints,
    Detection,
    DetectionResult,
    ImageResultWithSoftPrediction,
    InstanceSegmentationResult,
    PredictedMask,
    SegmentedObject,
    SegmentedObjectWithRects,
    VisualPromptingResult,
    ZSLVisualPromptingResult,
)

__all__ = [
    "AnomalyResult",
    "ClassificationResult",
    "Contour",
    "Detection",
    "DetectionResult",
    "DetectedKeypoints",
    "SegmentedObject",
    "SegmentedObjectWithRects",
    "ImageResultWithSoftPrediction",
    "InstanceSegmentationResult",
    "PredictedMask",
    "VisualPromptingResult",
    "ZSLVisualPromptingResult",
]
