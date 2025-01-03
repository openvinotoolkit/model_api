"""Tests for primitives."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import PIL

from model_api.visualizer import Overlay


def test_overlay(mock_image: PIL.Image):
    """Test if the overlay is created correctly."""
    empty_image = PIL.Image.new("RGB", (100, 100))
    expected_image = PIL.Image.blend(empty_image, mock_image, 0.4)
    # Test from image
    overlay = Overlay(mock_image)
    assert overlay.compute(empty_image) == expected_image

    # Test from numpy array
    data = np.zeros((100, 100, 3), dtype=np.uint8)
    data *= 255
    overlay = Overlay(data)
    assert overlay.compute(empty_image) == expected_image
