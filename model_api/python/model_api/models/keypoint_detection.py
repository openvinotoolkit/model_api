"""Definition for anomaly models.

Note: This file will change when anomalib is upgraded in OTX. CVS-114640

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

from .image_model import ImageModel
from .types import ListValue
from .utils import DetectedKeypoints


class KeypointDetection(ImageModel):
    __model__ = "keypoint_detection"

    def __init__(self, inference_adapter, configuration=dict(), preload=False):
        super().__init__(inference_adapter, configuration, preload)
        self._check_io_number(1, 2)
        simcc_split_ratio = 2.0
        sigma = (4.9, 5.66)
        decoder_cfg= {
            "input_size": self.input_size,
            "simcc_split_ratio": simcc_split_ratio,
            "sigma": sigma,
            }
        self.kp_dencoder = SimCCLabel(**decoder_cfg)

    def postprocess(self, outputs: dict[str, np.ndarray], meta: dict[str, Any]) -> DetectedKeypoints:
        encoded_kps = list(outputs.values())
        batch_keypoints, batch_scores = self.kp_dencoder.decode(*encoded_kps)
        orig_h, orig_w = meta["original_shape"][:2]
        model_w, model_h = tuple(self.input_size)
        kp_scale_h = orig_h / model_h
        kp_scale_w = orig_w / model_w
        batch_keypoints = batch_keypoints.squeeze() * np.array([kp_scale_w, kp_scale_h])
        return DetectedKeypoints(batch_keypoints, batch_scores.squeeze())

    @classmethod
    def parameters(cls) -> dict:
        parameters = super().parameters()
        parameters.update(
            {
                "input_size": ListValue(description="List of class labels", value_type=int, default_value=[256, 192]),
            }
        )
        return parameters


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
        input_size (tuple): Input image size in [h, w]
        smoothing_type (str): The SimCC label smoothing strategy. Options are
            ``'gaussian'`` and ``'standard'``. Defaults to ``'gaussian'``
        sigma (float | int | tuple): The sigma value in the Gaussian SimCC
            label. Defaults to 6.0
        simcc_split_ratio (float): The ratio of the label size to the input
            size. For example, if the input width is ``w``, the x label size
            will be :math:`w*simcc_split_ratio`. Defaults to 2.0
        label_smooth_weight (float): Label Smoothing weight. Defaults to 0.0
        decode_beta (float): The beta value for decoding visibility. Defaults
            to 150.0.

    .. _`SimCC: a Simple Coordinate Classification Perspective for Human Pose
    Estimation`: https://arxiv.org/abs/2107.03332
    """

    def __init__(
        self,
        input_size: tuple[int, int],
        smoothing_type: str = "gaussian",
        sigma: float | int | tuple[float, float] = 6.0,
        simcc_split_ratio: float = 2.0,
        label_smooth_weight: float = 0.0,
        decode_beta: float = 150.0,
    ) -> None:
        self.input_size = input_size
        self.smoothing_type = smoothing_type
        self.simcc_split_ratio = simcc_split_ratio
        self.label_smooth_weight = label_smooth_weight
        self.decode_beta = decode_beta

        if isinstance(sigma, (float, int)):
            self.sigma = np.array([sigma, sigma])
        else:
            self.sigma = np.array(sigma)

        if self.smoothing_type not in {"gaussian", "standard"}:
            msg = f"{self.__class__.__name__} got invalid `smoothing_type` value {self.smoothing_type}."
            raise ValueError(msg)

        if self.smoothing_type == "gaussian" and self.label_smooth_weight > 0:
            msg = "Attribute `label_smooth_weight` is only used for `standard` mode."
            raise ValueError(msg)

        if self.label_smooth_weight < 0.0 or self.label_smooth_weight > 1.0:
            msg = "`label_smooth_weight` should be in range [0, 1]."
            raise ValueError(msg)

    def decode(self, simcc_x: np.ndarray, simcc_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    apply_softmax: bool = False,
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
        apply_softmax (bool): whether to apply softmax on the heatmap.
            Defaults to False.

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

    if apply_softmax:
        simcc_x = simcc_x - np.max(simcc_x, axis=1, keepdims=True)
        simcc_y = simcc_y - np.max(simcc_y, axis=1, keepdims=True)
        ex, ey = np.exp(simcc_x), np.exp(simcc_y)
        simcc_x = ex / np.sum(ex, axis=1, keepdims=True)
        simcc_y = ey / np.sum(ey, axis=1, keepdims=True)

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
