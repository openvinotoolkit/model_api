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

from typing import Any

import numpy as np

from model_api.models import SAMImageEncoder, SAMDecoder
from model_api.models.utils import VisualPromptingResult


class SAMVisualPrompter:
    def __init__(
        self,
        encoder_model: SAMImageEncoder,
        decoder_model: SAMDecoder,
    ):
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model

    def infer(
        self,
        image: np.ndarray,
        boxes: np.ndarray | None,
        points: np.ndarray | None,
        labels: dict[str, np.ndarray] | None,
    ) -> VisualPromptingResult:
        outputs: list[dict[str, Any]] = []

        processed_image, meta = self.encoder_model.preprocess(image)
        image_embeddings = self.encoder_model.infer_sync(processed_image)
        processed_prompts = self.decoder_model.preprocess(
            {
                "bboxes": boxes,
                "points": points,
                "labels": labels,
                "orig_size": meta["original_shape"][:2],
            },
        )

        for prompt in processed_prompts:
            label = prompt.pop("label")
            prompt.update(**image_embeddings)

            prediction = self.decoder_model.infer_sync(prompt)
            prediction["scores"] = prediction["iou_predictions"]
            prediction["labels"] = label
            processed_prediction = self.decoder_model.postprocess(prediction, meta)
            outputs.append(processed_prediction)

        return VisualPromptingResult(
            upscaled_masks=[item["upscaled_masks"] for item in outputs],
            low_res_masks=[item["low_res_masks"] for item in outputs],
            iou_predictions=[item["iou_predictions"] for item in outputs],
            scores=[item["scores"] for item in outputs],
            labels=[item["labels"] for item in outputs],
            hard_predictions=[item["hard_prediction"] for item in outputs],
            soft_predictions=[item["soft_prediction"] for item in outputs],
        )

    def __call__(self,
                 image: np.ndarray,
                 boxes: np.ndarray | None,
                 points: np.ndarray | None,
                 labels: dict[str, np.ndarray] | None,
    ) -> VisualPromptingResult:
        return self.infer(image, boxes, points, labels)


class SAMLearnableVisualPrompter(SAMVisualPrompter):
    def learn(self, image, prompts, reset_ref_featires: bool = False):
        if reset_ref_featires or self.ref_embeddings_state is None:
            self._reset_inference_features()

    def infer(self, image, reference_features):
        pass
