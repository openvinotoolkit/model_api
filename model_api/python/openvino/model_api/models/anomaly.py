"""Definition for anomaly models.

Note: This file will change when anomalib is upgraded in OTX. CVS-114640
"""

"""
 Copyright (c) 2021-2023 Intel Corporation

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
from .types import ListValue, NumericalValue, StringValue
from .image_model import ImageModel
from .utils import AnomalyResult
import cv2


class AnomalyDetection(ImageModel):
    __model__ = "AnomalyDetection"

    def __init__(self, inference_adapter, configuration=None, preload=False):
        super().__init__(inference_adapter, configuration, preload)
        self.output_name = self.inference_adapter.model.outputs[0].any_name
        # attributes for mypy
        self.max: float
        self.min: float
        self.image_threshold: float
        self.pixel_threshold: float
        self.task: str

    def preprocess(self, inputs: np.ndarray):
        inputs = inputs / 255.0  # model expects inputs in range [0, 1]
        return super().preprocess(inputs)

    def postprocess(self, outputs: dict[str, np.ndarray], meta: dict[str, Any]):
        """Post-processes the outputs and returns the results.

        Args:
            outputs (dict[str, np.ndarray]): Raw model outputs
            meta (dict[str, Any]): Meta data containing the original image shape

        Returns:
            _type_: Results
        """
        anomaly_map: np.ndarray | None = None
        pred_label: str | None = None
        pred_mask: np.ndarray | None = None
        pred_boxes: np.ndarray | None = None
        box_labels: np.ndarray | None = None

        predictions = outputs[self.output_name]

        if len(predictions.shape) == 1:
            pred_score = predictions
        else:
            anomaly_map = predictions.squeeze()
            pred_score = anomaly_map.reshape(-1).max()

        if hasattr(self, "image_threshold"):
            pred_label = "Anomalous" if pred_score > self.image_threshold else "Normal"

        if hasattr(self, "task") and self.task in ("segmentation", "detection"):
            assert anomaly_map is not None  # for mypy
            pred_mask = (anomaly_map >= self.pixel_threshold).astype(np.uint8)
            anomaly_map = self._normalize(anomaly_map, self.pixel_threshold)

        # normalize
        pred_score = self._normalize(pred_score, self.image_threshold)

        # resize outputs
        if anomaly_map is not None:
            anomaly_map = cv2.resize(anomaly_map, (meta["original_shape"][1], meta["original_shape"][0]))
            pred_mask = cv2.resize(pred_mask, (meta["original_shape"][1], meta["original_shape"][0]))

        if hasattr(self, "task") and self.task == "detection":
            pred_boxes = self._get_boxes(pred_mask)
            box_labels = np.ones(pred_boxes.shape[0])

        return AnomalyResult(
            anomaly_map=anomaly_map,
            box_labels=box_labels,
            pred_boxes=pred_boxes,
            pred_label=pred_label,
            pred_mask=pred_mask,
            pred_score=pred_score.item(),
        )

    @classmethod
    def parameters(cls) -> dict:
        parameters = super().parameters()
        parameters.update(
            {
                "image_shape": ListValue(
                    description="Image shape",
                ),
                "image_threshold": NumericalValue(description="Image threshold", min=0.0, default_value=0.5),
                "pixel_threshold": NumericalValue(description="Pixel threshold", min=0.0, default_value=0.5),
                "max": NumericalValue(
                    description="max value for normalization",
                ),
                "min": NumericalValue(
                    description="min value for normalization",
                ),
                "task": StringValue(description="Task type", default_value="segmentation"),
            }
        )
        return parameters

    def _normalize(self, tensor: np.ndarray, threshold: float) -> np.ndarray:
        """Currently supports only min-max normalization."""
        normalized = ((tensor - threshold) / (self.max - self.min)) + 0.5
        normalized = np.clip(normalized, 0, 1)
        return normalized

    @staticmethod
    def _get_boxes(mask: np.ndarray) -> np.ndarray:
        """Get bounding boxes from mask.

        Args:
            mask (np.ndarray): Input mask of shapw (H, W)

        Returns:
            np.ndarray: array of shape (N,4) containing the bounding box coordinates of the objects in the masks in
                format [x1, y1, x2, y2]
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append([x, y, x + w, y + h])
        return np.array(boxes)
