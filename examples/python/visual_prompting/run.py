#!/usr/bin/env python3
"""
 Copyright (C) 2024 Intel Corporation

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

import argparse

import cv2
import numpy as np
import colorsys

from model_api.models import Model, SAMVisualPrompter, Prompt


def get_colors(n: int):
    HSV_tuples = [(x/n, 0.5, 0.5) for x in range(n)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    return (np.array(list(RGB_tuples))*255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="SAM sample script")
    parser.add_argument("image", type=str)
    parser.add_argument("encoder_path", type=str)
    parser.add_argument("decoder_path", type=str)
    parser.add_argument("prompts", nargs="+", type=int)
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise RuntimeError("Failed to read the image")

    encoder = Model.create_model(args.encoder_path)
    decoder = Model.create_model(args.decoder_path)
    sam_prompter = SAMVisualPrompter(encoder, decoder)

    all_prompts = []
    for i, p in enumerate(np.array(args.prompts).reshape(-1,2)):
        all_prompts.append(Prompt(p, i))

    result = sam_prompter(image, points=all_prompts)

    colors = get_colors(len(all_prompts))

    for i in range(len(result.upscaled_masks)):
        mask, iou = result.get_aggregated_hard_mask(i)
        print(f"Mask score {iou:.3f} for prompt {i}")
        masked_img = np.where(mask[...,None], colors[i], image)
        image = cv2.addWeighted(image, 0.2, masked_img, 0.8, 0)

    cv2.imwrite("sam_result.jpg", image)

#python examples/python/visual_prompting/run.py ./data/coco128/images/train2017/000000000127.jpg ./data/otx_models/sam_vit_b_zsl_encoder.xml ./data/otx_models/sam_vit_b_zsl_decoder.xml  274 306 482 295

if __name__ == "__main__":
    main()
