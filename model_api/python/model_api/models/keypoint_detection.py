"""
 Copyright (c) 2024 Intel Corporation

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

from __future__ import annotations

from typing import Any

import numpy as np
from model_api.pipelines import AsyncPipeline

from .image_model import ImageModel
from .types import ListValue
from .utils import DetectedKeypoints, Detection


class KeypointDetectionModel(ImageModel):
    """
    A wrapper that implements a basic keypoint regression model.
    """

    __model__ = "keypoint_detection"

    def __init__(self, inference_adapter, configuration=dict(), preload=False):
        """
        Initializes the keypoint detection model.

        Args:
            inference_adapter (InferenceAdapter): inference adapter containing the underlying model.
            configuration (dict, optional): Configuration overrides the model parameters (see parameters() method).
              Defaults to dict().
            preload (bool, optional): Forces inference adapter to load the model. Defaults to False.
        """
        super().__init__(inference_adapter, configuration, preload)
        self._check_io_number(1, 2)
        self.kp_dencoder = SimCCLabel()

    def postprocess(
        self, outputs: dict[str, np.ndarray], meta: dict[str, Any]
    ) -> DetectedKeypoints:
        """
        Applies SCC decoded to the model outputs.

        Args:
            outputs (dict[str, np.ndarray]): raw outputs of the model
            meta (dict[str, Any]): meta information about the input data

        Returns:
            DetectedKeypoints: detected keypoints
        """
        encoded_kps = list(outputs.values())
        batch_keypoints, batch_scores = self.kp_dencoder.decode(*encoded_kps)
        orig_h, orig_w = meta["original_shape"][:2]
        kp_scale_h = orig_h / self.h
        kp_scale_w = orig_w / self.w
        batch_keypoints = batch_keypoints.squeeze() * np.array([kp_scale_w, kp_scale_h])
        return DetectedKeypoints(batch_keypoints, batch_scores.squeeze())

    @classmethod
    def parameters(cls) -> dict:
        parameters = super().parameters()
        parameters.update(
            {
                "labels": ListValue(
                    description="List of class labels", value_type=str, default_value=[]
                ),
            }
        )
        return parameters


class TopDownKeypointDetectionPipeline:
    """
    Pipeline implementing top down keypoint detection approach.
    """

    def __init__(self, base_model: KeypointDetectionModel) -> None:
        self.base_model = base_model
        self.async_pipeline = AsyncPipeline(self.base_model)

    def predict(
        self, image: np.ndarray, detections: list[Detection]
    ) -> list[DetectedKeypoints]:
        """
        Predicts keypoints for the given image and detections.

        Args:
            image (np.ndarray): input full-size image
            detections (list[Detection]): detections located within the given image

        Returns:
            list[DetectedKeypoints]: per detection keypoints in detection coordinates
        """
        crops = []
        for det in detections:
            crops.append(image[det.ymin : det.ymax, det.xmin : det.xmax])

        crops_results = self.predict_crops(crops)
        for i, det in enumerate(detections):
            crops_results[i] = DetectedKeypoints(
                crops_results[i].keypoints + np.array([det.xmin, det.ymin]),
                crops_results[i].scores,
            )

        return crops_results

    def predict_crops(self, crops: list[np.ndarray]) -> list[DetectedKeypoints]:
        """
        Predicts keypoints for the given crops.

        Args:
            crops (list[np.ndarray]): list of cropped object images

        Returns:
            list[DetectedKeypoints]: per crop keypoints
        """
        for i, crop in enumerate(crops):
            self.async_pipeline.submit_data(crop, i)
        self.async_pipeline.await_all()

        num_crops = len(crops)
        result = []
        for j in range(num_crops):
            crop_prediction, _ = self.async_pipeline.get_result(j)
            result.append(crop_prediction)

        return result


class SimCCLabel:
    """Generate keypoint representation via "SimCC" approach.

    See the paper: `SimCC: a Simple Coordinate Classification Perspective for
    Human Pose Estimation`_ by Li et al (2022) for more details.
    Old name: SimDR

    Note:
        - instance number: N
        - keypoint number: K
        - keypoint dimension: D
        - image size: [h, w]

    Encoded:

        - keypoint_x_labels (np.ndarray): The generated SimCC label for x-axis.
            The label shape is (N, K, Wx) if ``smoothing_type=='gaussian'``
            and (N, K) if `smoothing_type=='standard'``, where
            :math:`Wx=w*simcc_split_ratio`
        - keypoint_y_labels (np.ndarray): The generated SimCC label for y-axis.
            The label shape is (N, K, Wy) if ``smoothing_type=='gaussian'``
            and (N, K) if `smoothing_type=='standard'``, where
            :math:`Wy=h*simcc_split_ratio`
        - keypoint_weights (np.ndarray): The target weights in shape (N, K)

    Args:
        simcc_split_ratio (float): The ratio of the label size to the input
            size. For example, if the input width is ``w``, the x label size
            will be :math:`w*simcc_split_ratio`. Defaults to 2.0

    .. _`SimCC: a Simple Coordinate Classification Perspective for Human Pose
    Estimation`: https://arxiv.org/abs/2107.03332
    """

    def __init__(
        self,
        smoothing_type: str = "gaussian",
        simcc_split_ratio: float = 2.0,
    ) -> None:
        self.simcc_split_ratio = simcc_split_ratio

    def decode(
        self, simcc_x: np.ndarray, simcc_y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decode keypoint coordinates from SimCC representations. The decoded coordinates are in the input image space.

        Args:
            encoded (Tuple[np.ndarray, np.ndarray]): SimCC labels for x-axis
                and y-axis
            simcc_x (np.ndarray): SimCC label for x-axis
            simcc_y (np.ndarray): SimCC label for y-axis

        Returns:
            tuple:
            - keypoints (np.ndarray): Decoded coordinates in shape (N, K, D)
            - socres (np.ndarray): The keypoint scores in shape (N, K).
                It usually represents the confidence of the keypoint prediction
        """
        keypoints, scores = get_simcc_maximum(simcc_x, simcc_y)

        # Unsqueeze the instance dimension for single-instance results
        if keypoints.ndim == 2:
            keypoints = keypoints[None, :]
            scores = scores[None, :]

        keypoints /= self.simcc_split_ratio

        return keypoints, scores


def get_simcc_maximum(
    simcc_x: np.ndarray,
    simcc_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Get maximum response location and value from simcc representations.

    Note:
        instance number: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        simcc_x (np.ndarray): x-axis SimCC in shape (K, Wx) or (N, K, Wx)
        simcc_y (np.ndarray): y-axis SimCC in shape (K, Wy) or (N, K, Wy)

    Returns:
        tuple:
        - locs (np.ndarray): locations of maximum heatmap responses in shape
            (K, 2) or (N, K, 2)
        - vals (np.ndarray): values of maximum heatmap responses in shape
            (K,) or (N, K)
    """
    if simcc_x.ndim not in (2, 3):
        msg = f"Invalid shape {simcc_x.shape}"
        raise ValueError(msg)
    if simcc_y.ndim not in (2, 3):
        msg = f"Invalid shape {simcc_y.shape}"
        raise ValueError(msg)
    if simcc_x.ndim != simcc_y.ndim:
        msg = f"{simcc_x.shape} != {simcc_y.shape}"
        raise ValueError(msg)

    if simcc_x.ndim == 3:
        batch_size, num_keypoints, _ = simcc_x.shape
        simcc_x = simcc_x.reshape(batch_size * num_keypoints, -1)
        simcc_y = simcc_y.reshape(batch_size * num_keypoints, -1)
    else:
        batch_size = None

    x_locs = np.argmax(simcc_x, axis=1)
    y_locs = np.argmax(simcc_y, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    max_val_x = np.amax(simcc_x, axis=1)
    max_val_y = np.amax(simcc_y, axis=1)

    mask = max_val_x > max_val_y
    max_val_x[mask] = max_val_y[mask]
    vals = max_val_x
    locs[vals <= 0.0] = -1

    if batch_size:
        locs = locs.reshape(batch_size, num_keypoints, 2)
        vals = vals.reshape(batch_size, num_keypoints)

    return locs, vals
