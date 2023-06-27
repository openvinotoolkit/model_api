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
from openvino.model_api.models.types import NumericalValue
from openvino.model_api.models.utils import Detection, DetectionResult, nms

from .tiler import Tiler


class DetectionTiler(Tiler):
    """
    Tiler for object detection models.
    This tiler expects model to output a lsit of `Detection` objects
    or one `DetectionResult` object.
    """

    def __init__(self, model, configuration=None, execution_mode="async"):
        super().__init__(model, configuration, execution_mode)

    @classmethod
    def parameters(cls):
        """Defines the description and type of configurable data parameters for the tiler.

        Returns:
             - the dictionary with defined wrapper tiler parameters
        """
        parameters = super().parameters()
        parameters.update(
            {
                "max_pred_number": NumericalValue(
                    value_type=int,
                    default_value=100,
                    min=1,
                    description="Maximum numbers of prediction per image",
                ),
            }
        )
        return parameters

    def _postprocess_tile(self, predictions, meta):
        """Converts predictions to a format convinient for further merging.

        Args:
             predictions: predictions from a detection model: a list of `Detection` objects
             or one `DetectionResult`
             meta: a dict containing key "coord", representing tile coordinates

        Returns:
             a dict with postprocessed predictions in 6-items format: (label id, score, bbox)
        """

        output_dict = {}
        if hasattr(predictions, "objects"):
            detections = _detection2array(predictions.objects)
            output_dict["saliency_map"] = predictions.saliency_map
            output_dict["features"] = predictions.feature_vector
        else:
            detections = _detection2array(predictions)

        offset_x, offset_y = meta["coord"][:2]
        detections[:, 2:] += np.tile((offset_x, offset_y), 2)
        output_dict["bboxes"] = detections

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

        detections_array = np.empty((0, 6), dtype=np.float32)
        for result in results:
            if len(result["bboxes"]):
                detections_array = np.concatenate((detections_array, result["bboxes"]))

        if np.prod(detections_array.shape):
            detections_array, keep_idx = _multiclass_nms(
                detections_array, max_num=self.max_pred_number
            )
            if meta is not None:
                meta["keep_idx"] = keep_idx

        detected_objects = []
        for i in range(detections_array.shape[0]):
            label = int(detections_array[i][0])
            score = float(detections_array[i][1])
            bbox = list(detections_array[i][2:])
            detected_objects.append(
                Detection(*bbox, score, label, self.model.labels[label])
            )

        return DetectionResult(
            detected_objects,
            np.ndarray(0),
            np.ndarray(0),
        )


def _multiclass_nms(
    detections,
    iou_threshold=0.45,
    max_num=200,
):
    """Multi-class NMS.

    strategy: in order to perform NMS independently per class,
    we add an offset to all the boxes. The offset is dependent
    only on the class idx, and is large enough so that boxes
    from different classes do not overlap

    Args:
        detections (np.ndarray): labels, scores and boxes
        iou_threshold (float, optional): IoU threshold. Defaults to 0.45.
        max_num (int, optional): Max number of objects filter. Defaults to 200.

    Returns:
        tuple: (dets, indices), Dets are boxes with scores. Indices are indices of kept boxes.
    """
    labels = detections[:, 0]
    scores = detections[:, 1]
    boxes = detections[:, 2:]
    max_coordinate = boxes.max()
    offsets = labels.astype(boxes.dtype) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]

    keep = nms(*boxes_for_nms.T, scores, iou_threshold)
    if max_num > 0:
        keep = keep[:max_num]
    keep = np.array(keep)
    det = detections[keep]
    return det, keep


def _detection2array(detections):
    """Convert list of OpenVINO Detection to a numpy array.

    Args:
        detections (List): List of OpenVINO Detection containing score, id, xmin, ymin, xmax, ymax

    Returns:
        np.ndarray: numpy array with [label, confidence, x1, y1, x2, y2]
    """
    scores = np.empty((0, 1), dtype=np.float32)
    labels = np.empty((0, 1), dtype=np.uint32)
    boxes = np.empty((0, 4), dtype=np.float32)
    for det in detections:
        if (det.xmax - det.xmin) * (det.ymax - det.ymin) < 1.0:
            continue
        scores = np.append(scores, [[det.score]], axis=0)
        labels = np.append(labels, [[det.id]], axis=0)
        boxes = np.append(
            boxes,
            [[float(det.xmin), float(det.ymin), float(det.xmax), float(det.ymax)]],
            axis=0,
        )
    detections = np.concatenate((labels, scores, boxes), -1)
    return detections
