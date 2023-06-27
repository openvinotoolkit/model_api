"""
 Copyright (c) 2023 Intel Corporation

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

import numpy as np
from openvino.model_api.models.instance_segmentation import _segm_postprocess
from openvino.model_api.models.utils import SegmentedObject

from .detection import DetectionTiler


class InstanceSegmentationTiler(DetectionTiler):
    """
    Tiler for object instance segmentation models.
    This tiler expects model to output a lsit of `SegmentedObject` objects.

    In addition, this tiler allows to use a tile classifier model,
    which predicts objectness score for each tile. Later, tiles can
    be filtered by this score.
    """

    def __init__(
        self,
        model,
        configuration=None,
        execution_mode="async",
        tile_classifier_model=None,
    ):
        """
        Constructor for creating a semantic segmentation tiling pipeline

        Args:
            model: underlying model
            configuration: it contains values for parameters accepted by specific
            tiler (`tile_size`, `tiles_overlap` etc.) which are set as data attributes.
            execution_mode: Controls inference mode of the tiler (`async` or `sync`).
            tile_classifier_model: an `ImageModel`, which has "tile_prob" output.
        """
        super().__init__(model, configuration, execution_mode)
        self.tile_classifier_model = tile_classifier_model

    def _filter_tiles(self, image, tile_coords, confidence_threshold=0.35):
        """Filter tiles by objectness score provided by tile classifier

        Args:
             image: full size image
             tile_coords: tile coordinates

        Returns:
             tile coordinates to keep
        """
        if self.tile_classifier_model is not None:
            keep_coords = []
            for i, coord in enumerate(tile_coords):
                tile_img = self._crop_tile(image, coord)
                tile_dict, _ = self.model.preprocess(tile_img)
                cls_outputs = self.tile_classifier_model.infer_sync(tile_dict)
                if i == 0 or cls_outputs["tile_prob"] > confidence_threshold:
                    keep_coords.append(coord)
            return keep_coords

        return tile_coords

    def _postprocess_tile(self, predictions, meta):
        """Converts predictions to a format convinient for further merging.

        Args:
             predictions: predictions from an instance segmentation model: a list of `SegmentedObject` objects
             meta: a dict containing key "coord", representing tile coordinates

        Returns:
             a dict with postprocessed detections in 6-items format: (label id, score, bbox) and masks
        """

        output_dict = super()._postprocess_tile(predictions, meta)
        if hasattr(predictions, "mask"):
            output_dict["masks"] = predictions.mask
        else:
            output_dict["masks"] = []
            for segm_res in predictions:
                output_dict["masks"].append(segm_res.mask)

        return output_dict

    def _merge_results(self, results, shape, meta=None):
        """Merge results from all tiles.

        To merge detections, per-class NMS is applied.

        Args:
             results: list of per-tile results
             shape: original full-res image shape
        Returns:
             merged prediciton
        """

        if meta is None:
            meta = {}
        detection_result = super()._merge_results(results, shape, meta)

        masks = []
        for result in results:
            if len(result["bboxes"]):
                masks.extend(result["masks"])

        if masks:
            masks = [masks[keep_idx] for keep_idx in meta["keep_idx"]]

            for i, (det, mask) in enumerate(zip(detection_result.objects, masks)):
                box = np.array([det.xmin, det.ymin, det.xmax, det.ymax])
                masks[i] = _segm_postprocess(box, mask, *shape[:-1])

        return [
            SegmentedObject(
                detection.xmin,
                detection.ymin,
                detection.xmax,
                detection.ymax,
                detection.score,
                detection.id,
                detection.str_label,
                mask,
            )
            for detection, mask in zip(detection_result.objects, masks)
        ]
