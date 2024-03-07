#!/usr/bin/env python3
"""
 Copyright (C) 2018-2022 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import sys

import cv2
from model_api.models import DetectionModel


def main():
    if len(sys.argv) != 2:
        raise RuntimeError(f"Usage: {sys.argv[0]} <path_to_image>")

    image = cv2.imread(sys.argv[1])
    if image is None:
        raise RuntimeError("Failed to read the image")

    # Create Object Detection model specifying the OVMS server URL
    model = DetectionModel.create_model(
        "localhost:9000/models/ssd_mobilenet_v1_fpn_coco", model_type="ssd"
    )
    detections = model(image)
    print(f"Detection results: {detections}")


if __name__ == "__main__":
    main()
