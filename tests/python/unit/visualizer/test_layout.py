"""Test layout."""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from PIL import Image

from model_api.visualizer import Flatten, Scene
from model_api.visualizer.primitive import Overlay


def test_flatten_layout(mock_image: Image, mock_scene: Scene):
    """Test if the layout is created correctly."""
    overlay = np.zeros((100, 100, 3), dtype=np.uint8)
    overlay[50, 50] = [255, 0, 0]
    overlay = Image.fromarray(overlay)

    expected_image = Image.blend(mock_image, overlay, 0.4)
    mock_scene.layout = Flatten(Overlay)
    assert mock_scene.render() == expected_image
