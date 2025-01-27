"""Visualization Example."""

# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
from argparse import Namespace

import cv2
import numpy as np
from PIL import Image

from model_api.models import Model
from model_api.visualizer import Visualizer


def main(args: Namespace):
    image = Image.open(args.image)

    model = Model.create_model(args.model)

    image_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    predictions = model(image_array)
    visualizer = Visualizer()

    if args.output:
        visualizer.save(image=image, result=predictions, path=args.output)
    else:
        visualizer.show(image=image, result=predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, required=False)
    args = parser.parse_args()
    main(args)
