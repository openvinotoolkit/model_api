"""Tests for scene."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
from PIL import Image

from model_api.models.result import AnomalyResult, ClassificationResult, DetectionResult
from model_api.models.result.classification import Label
from model_api.visualizer import Visualizer


def test_anomaly_scene(mock_image: Image, tmpdir: Path):
    """Test if the anomaly scene is created."""
    heatmap = np.ones(mock_image.size, dtype=np.uint8)
    heatmap *= 255

    mask = np.zeros(mock_image.size, dtype=np.uint8)
    mask[32:96, 32:96] = 255
    mask[40:80, 0:128] = 255

    anomaly_result = AnomalyResult(
        anomaly_map=heatmap,
        pred_boxes=np.array([[0, 0, 128, 128], [32, 32, 96, 96]]),
        pred_label="Anomaly",
        pred_mask=mask,
        pred_score=0.85,
    )

    visualizer = Visualizer()
    visualizer.save(mock_image, anomaly_result, tmpdir / "anomaly_scene.jpg")
    assert Path(tmpdir / "anomaly_scene.jpg").exists()


def test_classification_scene(mock_image: Image, tmpdir: Path):
    """Test if the classification scene is created."""
    classification_result = ClassificationResult(
        top_labels=[
            Label(name="cat", confidence=0.95),
            Label(name="dog", confidence=0.90),
        ],
        saliency_map=np.ones(mock_image.size, dtype=np.uint8),
    )
    visualizer = Visualizer()
    visualizer.save(
        mock_image, classification_result, tmpdir / "classification_scene.jpg"
    )
    assert Path(tmpdir / "classification_scene.jpg").exists()


def test_detection_scene(mock_image: Image, tmpdir: Path):
    """Test if the detection scene is created."""
    detection_result = DetectionResult(
        bboxes=np.array([[0, 0, 128, 128], [32, 32, 96, 96]]),
        labels=np.array([1, 2]),
        label_names=["person", "car"],
        scores=np.array([0.85, 0.75]),
        saliency_map=(np.ones((1, 2, 6, 8)) * 255).astype(np.uint8),
    )
    visualizer = Visualizer()
    visualizer.save(mock_image, detection_result, tmpdir / "detection_scene.jpg")
    assert Path(tmpdir / "detection_scene.jpg").exists()
