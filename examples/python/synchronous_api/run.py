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
from openvino.model_api.models import (
    ClassificationModel,
    DetectionModel,
    SegmentationModel,
)
from PIL import Image


def main():
    if len(sys.argv) != 2:
        raise RuntimeError(f"Usage: {sys.argv[0]} <path_to_image>")

    image = cv2.imread(sys.argv[1])
    if image is None:
        raise RuntimeError("Failed to read the image")

    # Create Image Classification model using mode name and download from Open Model Zoo
    efficientnet_b0 = ClassificationModel.create_model("efficientnet-b0-pytorch")
    classifications = efficientnet_b0(image)
    print(f"Classification results: {classifications}")

    # Create Object Detection model using mode name and download from Open Model Zoo
    # Replace numpy preprocessing and embed it directly into a model graph to speed up inference
    # download_dir is used to store downloaded model
    ssd_mobilenet_fpn = DetectionModel.create_model(
        "ssd_mobilenet_v1_fpn_coco",
        download_dir="tmp",
    )
    detections = ssd_mobilenet_fpn(image)
    print(f"Detection results: {detections}")
    ssd_mobilenet_fpn.save("ssd_mobilenet_v1_fpn_coco_with_preprocessing.xml")

    # Instantiate from a local model (downloaded previously)
    ssd_mobilenet_fpn_local = DetectionModel.create_model(
        "tmp/public/ssd_mobilenet_v1_fpn_coco/FP16/ssd_mobilenet_v1_fpn_coco.xml"
    )
    detections = ssd_mobilenet_fpn_local(image)
    print(f"Detection results for local: {detections}")

    # Create Image Segmentation model
    hrnet = SegmentationModel.create_model("hrnet-v2-c1-segmentation")
    mask = hrnet(image)
    Image.fromarray(mask + 20).save("mask.png")


if __name__ == "__main__":
    main()
