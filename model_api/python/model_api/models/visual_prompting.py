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
from copy import deepcopy
from collections import defaultdict
from itertools import product

import numpy as np
import cv2

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

    def __call__(
        self,
        image: np.ndarray,
        boxes: np.ndarray | None,
        points: np.ndarray | None,
        labels: dict[str, np.ndarray] | None,
    ) -> VisualPromptingResult:
        return self.infer(image, boxes, points, labels)


class SAMLearnableVisualPrompter:
    def __init__(
        self,
        encoder_model: SAMImageEncoder,
        decoder_model: SAMDecoder,
        reference_features: np.ndarray | None = None,
    ):
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        self.reference_features = reference_features
        self.used_indices = None

        self.point_labels_box = np.array([[2, 3]], dtype=np.float32)
        self.has_mask_inputs = [np.array([[0.0]]), np.array([[1.0]])]

        self.is_cascade: bool = True
        self.threshold: float = 0.0
        self.num_bg_points: int = 1
        self.default_threshold_target: float = 0.65
        self.image_size: int = self.encoder_model.image_size
        self.downsizing: int = 64
        self.default_threshold_reference: float = 0.3

        if self.reference_features is None:
            self.reset_reference_info()

    def has_reference_features(self) -> bool:
        return self.reference_features is not None

    def learn(
        self,
        image: np.ndarray,
        boxes: np.ndarray | None,
        points: np.ndarray | None,
        labels: dict[str, np.ndarray] | None,
    ):
        processed_image, meta = self.encoder_model.preprocess(image)
        processed_prompts = self.decoder_model.preprocess(
            {
                "bboxes": boxes,
                "points": points,
                "labels": labels,
                "orig_size": meta["original_shape"][:2],
            },
        )

        processed_prompts_w_labels = self._gather_prompts_with_labels(processed_prompts)
        largest_label: int = max([int(p) for p in processed_prompts_w_labels] + [0])
        self._expand_reference_info(largest_label)

        original_shape = np.array(meta["original_shape"][:2])

        # forward image encoder
        image_embeddings = self.encoder_model.infer_sync(processed_image)
        processed_embedding = (
            image_embeddings["image_embeddings"].squeeze().transpose(1, 2, 0)
        )

        # get reference masks
        ref_masks: np.ndarray = np.zeros(
            (largest_label + 1, *original_shape), dtype=np.uint8
        )
        for label, input_prompts in processed_prompts_w_labels.items():
            ref_mask: np.ndarray = np.zeros(original_shape, dtype=np.uint8)
            for inputs_decoder in input_prompts:
                label = inputs_decoder.pop("label")  # noqa: PLW2901
                if "point_coords" in inputs_decoder:
                    # bboxes and points
                    inputs_decoder.update(image_embeddings)
                    prediction = self._predict_masks(
                        inputs_decoder, original_shape, is_cascade=self.is_cascade
                    )
                    masks = prediction["upscaled_masks"]
                else:
                    # log.warning("annotation and polygon will be supported.")
                    continue
                ref_mask[masks] += 1
            ref_mask = np.clip(ref_mask, 0, 1)

            ref_feat: np.ndarray | None = None
            cur_default_threshold_reference = deepcopy(self.default_threshold_reference)
            while ref_feat is None:
                # log.info(f"[*] default_threshold_reference : {cur_default_threshold_reference:.4f}")
                ref_feat = self._generate_masked_features(
                    feats=processed_embedding,
                    masks=ref_mask,
                    threshold_mask=cur_default_threshold_reference,
                    image_size=self.encoder_model.image_size,
                )
                cur_default_threshold_reference -= 0.05

            self.reference_feats[label] = ref_feat
            self.used_indices: np.ndarray = np.concatenate((self.used_indices, label))
            ref_masks[label] = ref_mask

        self.used_indices = np.unique(self.used_indices)

        return {
            "reference_feats": self.reference_feats,
            "used_indices": self.used_indices,
        }, ref_masks

    def reset_reference_info(self) -> None:
        """Initialize reference information."""
        self.reference_feats = np.zeros(
            (0, 1, self.decoder_model.embed_dim), dtype=np.float32
        )
        self.used_indices = np.array([], dtype=np.int64)

    def _gather_prompts_with_labels(
        self,
        image_prompts: list[dict[str, np.ndarray]],
    ) -> dict[int, list[dict[str, np.ndarray]]]:
        """Gather prompts according to labels."""

        processed_prompts: defaultdict[int, list[dict[str, np.ndarray]]] = defaultdict(
            list
        )
        for prompt in image_prompts:
            processed_prompts[int(prompt["label"])].append(prompt)

        return dict(sorted(processed_prompts.items(), key=lambda x: x))

    def _expand_reference_info(self, new_largest_label: int) -> None:
        """Expand reference info dimensions if newly given processed prompts have more lables."""
        if new_largest_label > (cur_largest_label := len(self.reference_feats) - 1):
            diff = new_largest_label - cur_largest_label
            self.reference_feats = np.pad(
                self.reference_feats, ((0, diff), (0, 0), (0, 0)), constant_values=0.0
            )

    def _generate_masked_features(
        self,
        feats: np.ndarray,
        masks: np.ndarray,
        threshold_mask: float,
        image_size: int = 1024,
    ) -> np.ndarray | None:
        """Generate masked features.

        Args:
            feats (np.ndarray): Raw reference features. It will be filtered with masks.
            masks (np.ndarray): Reference masks used to filter features.
            threshold_mask (float): Threshold to control masked region.
            image_size (int): Input image size.

        Returns:
            (np.ndarray): Masked features.
        """
        target_shape = image_size / max(masks.shape) * np.array(masks.shape)
        target_shape = target_shape[::-1].astype(np.int32)

        # Post-process masks
        masks = cv2.resize(masks, target_shape, interpolation=cv2.INTER_LINEAR)
        masks = self._pad_to_square(masks, image_size)
        masks = cv2.resize(masks, feats.shape[:2][::-1], interpolation=cv2.INTER_LINEAR)

        # Target feature extraction
        if (masks > threshold_mask).sum() == 0:
            # (for stability) there is no area to be extracted
            return None

        masked_feat = feats[masks > threshold_mask]
        masked_feat = masked_feat.mean(0)[None]
        return masked_feat / np.linalg.norm(masked_feat, axis=-1, keepdims=True)

    def _pad_to_square(self, x: np.ndarray, image_size: int = 1024) -> np.ndarray:
        """Pad to a square input.

        Args:
            x (np.ndarray): Mask to be padded.

        Returns:
            (np.ndarray): Padded mask.
        """
        h, w = x.shape[-2:]
        padh = image_size - h
        padw = image_size - w
        return np.pad(x, ((0, padh), (0, padw)), constant_values=0.0)

    def _predict_masks(
        self,
        inputs: dict[str, np.ndarray],
        original_size: np.ndarray,
        is_cascade: bool = False,
    ) -> dict[str, np.ndarray]:
        """Process function of OpenVINO Visual Prompting Inferencer."""
        masks: np.ndarray
        logits: np.ndarray
        scores: np.ndarray
        num_iter = 3 if is_cascade else 1
        for i in range(num_iter):
            if i == 0:
                # First-step prediction
                mask_input = np.zeros(
                    (1, 1, *(x * 4 for x in inputs["image_embeddings"].shape[2:])),
                    dtype=np.float32,
                )
                has_mask_input = self.has_mask_inputs[0]

            elif i == 1:
                # Cascaded Post-refinement-1
                mask_input, masks = self._decide_masks(
                    masks, logits, scores, is_single=True
                )  # noqa: F821
                if masks.sum() == 0:
                    return {"upscaled_masks": masks}

                has_mask_input = self.has_mask_inputs[1]

            elif i == 2:
                # Cascaded Post-refinement-2
                mask_input, masks = self._decide_masks(
                    masks, logits, scores
                )  # noqa: F821
                if masks.sum() == 0:
                    return {"upscaled_masks": masks}

                has_mask_input = self.has_mask_inputs[1]
                y, x = np.nonzero(masks)
                box_coords = self.decoder_model.apply_coords(
                    np.array(
                        [[x.min(), y.min()], [x.max(), y.max()]], dtype=np.float32
                    ),
                    original_size,
                )
                box_coords = np.expand_dims(box_coords, axis=0)
                inputs.update(
                    {
                        "point_coords": np.concatenate(
                            (inputs["point_coords"], box_coords), axis=1
                        ),
                        "point_labels": np.concatenate(
                            (inputs["point_labels"], self.point_labels_box), axis=1
                        ),
                    },
                )

            inputs.update({"mask_input": mask_input, "has_mask_input": has_mask_input})
            prediction = self.decoder_model.infer_sync(inputs)
            upscaled_masks, scores, logits = (
                prediction["upscaled_masks"],
                prediction["iou_predictions"],
                prediction["low_res_masks"],
            )
            masks = upscaled_masks > self.decoder_model.mask_threshold

        _, masks = self._decide_masks(masks, logits, scores)
        return {"upscaled_masks": masks}

    def _decide_masks(
        self,
        masks: np.ndarray,
        logits: np.ndarray,
        scores: np.ndarray,
        is_single: bool = False,
    ) -> tuple[np.ndarray, ...] | tuple[None, np.ndarray]:
        """Post-process logits for resized masks according to best index based on scores."""
        if is_single:
            best_idx = 0
        else:
            # skip the first index components
            scores, masks, logits = (x[:, 1:] for x in (scores, masks, logits))

            # filter zero masks
            while (
                len(scores[0]) > 0
                and masks[0, (best_idx := np.argmax(scores[0]))].sum() == 0
            ):
                scores, masks, logits = (
                    np.concatenate((x[:, :best_idx], x[:, best_idx + 1 :]), axis=1)
                    for x in (scores, masks, logits)
                )

            if len(scores[0]) == 0:
                # all predicted masks were zero masks, ignore them.
                return None, np.zeros(masks.shape[-2:])

            best_idx = np.argmax(scores[0])
        return logits[:, [best_idx]], masks[0, best_idx]

    def infer(
        self,
        image: np.ndarray,
        reference_features: np.ndarray | None,
        used_indices: np.ndarray | None,
    ):
        if reference_features is None:
            if self.reference_features is None:
                raise RuntimeError(
                    "Reference features are not defined. This parameter can be passed via SAMLearnableVisualPrompter constructor, or as an argument of infer() method"
                )
            else:
                reference_features = self.reference_features

        if used_indices is None:
            if self.used_indices is None:
                raise RuntimeError(
                    "Used indices are not defined. This parameter can be passed via SAMLearnableVisualPrompter constructor, or as an argument of infer() method"
                )
            else:
                used_indices = self.used_indices

        processed_image, meta = self.encoder_model.preprocess(image)
        original_shape = np.array(meta["original_shape"][:2])

        image_embeddings = self.encoder_model.infer_sync(processed_image)

        total_points_scores, total_bg_coords = self._get_prompt_candidates(
            image_embeddings=image_embeddings["image_embeddings"],
            reference_feats=reference_features,
            used_indices=used_indices,
            original_shape=original_shape,
            threshold=self.threshold,
            num_bg_points=self.num_bg_points,
            default_threshold_target=self.default_threshold_target,
            image_size=self.image_size,
            downsizing=self.downsizing,
        )

        predicted_masks: defaultdict[int, list] = defaultdict(list)
        used_points: defaultdict[int, list] = defaultdict(list)
        for label in total_points_scores:
            points_scores = total_points_scores[label]
            bg_coords = total_bg_coords[label]
            for points_score in points_scores:
                if points_score[-1] in [-1.0, 0.0]:
                    continue

                x, y = points_score[:2]
                is_done = False
                for pm in predicted_masks.get(label, []):
                    # check if that point is already assigned
                    if pm[int(y), int(x)] > 0:
                        is_done = True
                        break
                if is_done:
                    continue

                point_coords = np.concatenate(
                    (np.array([[x, y]]), bg_coords), axis=0, dtype=np.float32
                )
                point_coords = self.decoder_model.apply_coords(
                    point_coords, original_shape
                )
                point_labels = np.array([1] + [0] * len(bg_coords), dtype=np.float32)
                inputs_decoder = {
                    "point_coords": point_coords[None],
                    "point_labels": point_labels[None],
                    "orig_size": original_shape[None],
                }
                inputs_decoder.update(image_embeddings)

                prediction = self._predict_masks(
                    inputs_decoder, original_shape, self.is_cascade
                )
                prediction.update({"scores": points_score[-1]})

                predicted_masks[label].append(
                    prediction[self.decoder_model.output_blob_name]
                )
                used_points[label].append(points_score)

        # check overlapping area between different label masks
        self._inspect_overlapping_areas(predicted_masks, used_points)

        return (predicted_masks, used_points)

    def _get_prompt_candidates(
        self,
        image_embeddings: np.ndarray,
        reference_feats: np.ndarray,
        used_indices: np.ndarray,
        original_shape: np.ndarray,
        threshold: float = 0.0,
        num_bg_points: int = 1,
        default_threshold_target: float = 0.65,
        image_size: int = 1024,
        downsizing: int = 64,
    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        """Get prompt candidates."""
        target_feat = image_embeddings.squeeze()
        c_feat, h_feat, w_feat = target_feat.shape
        target_feat = target_feat / np.linalg.norm(target_feat, axis=0, keepdims=True)
        target_feat = target_feat.reshape(c_feat, h_feat * w_feat)

        total_points_scores: dict[int, np.ndarray] = {}
        total_bg_coords: dict[int, np.ndarray] = {}
        for label in used_indices:
            sim = reference_feats[label] @ target_feat
            sim = sim.reshape(h_feat, w_feat)
            sim = self._resize_to_original_shape(sim, image_size, original_shape)

            threshold = (threshold == 0) * default_threshold_target + threshold
            points_scores, bg_coords = self._point_selection(
                mask_sim=sim,
                original_shape=original_shape,
                threshold=threshold,
                num_bg_points=num_bg_points,
                image_size=image_size,
                downsizing=downsizing,
            )

            if points_scores is not None:
                total_points_scores[label] = points_scores
                total_bg_coords[label] = bg_coords
        return total_points_scores, total_bg_coords

    def _point_selection(
        self,
        mask_sim: np.ndarray,
        original_shape: np.ndarray,
        threshold: float = 0.0,
        num_bg_points: int = 1,
        image_size: int = 1024,
        downsizing: int = 64,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """Select point used as point prompts."""
        _, w_sim = mask_sim.shape

        # Top-first point selection
        point_coords = np.where(mask_sim > threshold)
        fg_coords_scores = np.stack(
            point_coords[::-1] + (mask_sim[point_coords],), axis=0
        ).T

        ## skip if there is no point coords
        if len(fg_coords_scores) == 0:
            return None, None

        ratio = image_size / original_shape.max()
        width = (original_shape[1] * ratio).astype(np.int64)
        n_w = width // downsizing

        ## get grid numbers
        idx_grid = (
            fg_coords_scores[:, 1] * ratio // downsizing * n_w
            + fg_coords_scores[:, 0] * ratio // downsizing
        )
        idx_grid_unique = np.unique(idx_grid.astype(np.int64))

        ## get matched indices
        matched_matrix = (
            np.expand_dims(idx_grid, axis=-1) == idx_grid_unique
        )  # (totalN, uniqueN)

        ## sample fg_coords_scores matched by matched_matrix
        matched_grid = np.expand_dims(fg_coords_scores, axis=1) * np.expand_dims(
            matched_matrix, axis=-1
        )

        ## sample the highest score one of the samples that are in the same grid
        matched_indices = self._topk_numpy(
            matched_grid[..., -1], k=1, axis=0, largest=True
        )[1][0].astype(np.int64)
        points_scores = matched_grid[matched_indices].diagonal().T

        ## sort by the highest score
        sorted_points_scores_indices = np.flip(
            np.argsort(points_scores[:, -1]), axis=-1
        ).astype(np.int64)
        points_scores = points_scores[sorted_points_scores_indices]

        # Top-last point selection
        bg_indices = self._topk_numpy(mask_sim.flatten(), num_bg_points, largest=False)[
            1
        ]
        bg_x = np.expand_dims(bg_indices // w_sim, axis=0)
        bg_y = bg_indices - bg_x * w_sim
        bg_coords = np.concatenate((bg_y, bg_x), axis=0).transpose(1, 0)
        bg_coords = bg_coords.astype(np.float32)

        return points_scores, bg_coords

    def _resize_to_original_shape(
        self, masks: np.ndarray, image_size: int, original_shape: np.ndarray
    ) -> np.ndarray:
        """Resize feature size to original shape."""
        # resize feature size to input size
        masks = cv2.resize(
            masks, (image_size, image_size), interpolation=cv2.INTER_LINEAR
        )

        # remove pad
        prepadded_size = self._get_prepadded_size(original_shape, image_size)
        masks = masks[..., : prepadded_size[0], : prepadded_size[1]]

        # resize unpadded one to original shape
        original_shape = original_shape.astype(np.int64)
        h, w = original_shape[0], original_shape[1]
        return cv2.resize(masks, (w, h), interpolation=cv2.INTER_LINEAR)

    def _get_prepadded_size(self, original_shape: int, image_size: int) -> np.ndarray:
        """Get pre-padded size."""
        scale = image_size / np.max(original_shape)
        transformed_size = scale * original_shape
        return np.floor(transformed_size + 0.5).astype(np.int64)

    def _topk_numpy(
        self, x: np.ndarray, k: int, axis: int = -1, largest: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Top-k function for numpy same with torch.topk."""
        if largest:
            k = -k
            indices = range(k, 0)
        else:
            indices = range(k)
        partitioned_ind = np.argpartition(x, k, axis=axis).take(
            indices=indices, axis=axis
        )
        partitioned_scores = np.take_along_axis(x, partitioned_ind, axis=axis)
        sorted_trunc_ind = np.argsort(partitioned_scores, axis=axis)
        if largest:
            sorted_trunc_ind = np.flip(sorted_trunc_ind, axis=axis)
        ind = np.take_along_axis(partitioned_ind, sorted_trunc_ind, axis=axis)
        scores = np.take_along_axis(partitioned_scores, sorted_trunc_ind, axis=axis)
        return scores, ind

    def _inspect_overlapping_areas(
        self,
        predicted_masks: dict[int, list[np.ndarray]],
        used_points: dict[int, list[np.ndarray]],
        threshold_iou: float = 0.8,
    ) -> None:
        def _calculate_mask_iou(
            mask1: np.ndarray, mask2: np.ndarray
        ) -> tuple[float, np.ndarray | None]:
            assert mask1.ndim == 2  # noqa: S101
            assert mask2.ndim == 2  # noqa: S101
            # Avoid division by zero
            if (union := np.logical_or(mask1, mask2).sum().item()) == 0:
                return 0.0, None
            intersection = np.logical_and(mask1, mask2)
            return intersection.sum().item() / union, intersection

        for (label, masks), (other_label, other_masks) in product(
            predicted_masks.items(), predicted_masks.items()
        ):
            if other_label <= label:
                continue

            overlapped_label = []
            overlapped_other_label = []
            for (im, mask), (jm, other_mask) in product(
                enumerate(masks), enumerate(other_masks)
            ):
                _mask_iou, _intersection = _calculate_mask_iou(mask, other_mask)
                if _mask_iou > threshold_iou:
                    if used_points[label][im][2] > used_points[other_label][jm][2]:
                        overlapped_other_label.append(jm)
                    else:
                        overlapped_label.append(im)
                elif _mask_iou > 0:
                    # refine the slightly overlapping region
                    overlapped_coords = np.where(_intersection)
                    if used_points[label][im][2] > used_points[other_label][jm][2]:
                        other_mask[overlapped_coords] = 0.0
                    else:
                        mask[overlapped_coords] = 0.0

            for im in sorted(set(overlapped_label), reverse=True):
                masks.pop(im)
                used_points[label].pop(im)

            for jm in sorted(set(overlapped_other_label), reverse=True):
                other_masks.pop(jm)
                used_points[other_label].pop(jm)
