#
# Copyright (C) 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np

from .detection_model import DetectionModel
from .result_types import BoxesLabelsParser, DetectionResult, MultipleOutputParser, SingleOutputParser

BBOX_AREA_THRESHOLD = 1.0
SALIENCY_MAP_NAME = "saliency_map"
FEATURE_VECTOR_NAME = "feature_vector"


class SSD(DetectionModel):
    __model__ = "SSD"

    def __init__(self, inference_adapter, configuration: dict = {}, preload=False):
        super().__init__(inference_adapter, configuration, preload)
        self.image_info_blob_name = self.image_info_blob_names[0] if len(self.image_info_blob_names) == 1 else None
        self.output_parser = self._get_output_parser(self.image_blob_name)

    def preprocess(self, inputs):
        dict_inputs, meta = super().preprocess(inputs)
        if self.image_info_blob_name:
            dict_inputs[self.image_info_blob_name] = np.array([[self.h, self.w, 1]])
        return dict_inputs, meta

    def postprocess(self, outputs, meta) -> DetectionResult:
        detections = self._parse_outputs(outputs)
        self._resize_detections(detections, meta)
        self._filter_detections(detections, BBOX_AREA_THRESHOLD)
        self._add_label_names(detections)
        detections.saliency_map = outputs.get(SALIENCY_MAP_NAME, np.ndarray(0))
        detections.feature_vector = outputs.get(FEATURE_VECTOR_NAME, np.ndarray(0))
        return detections

    def _get_output_parser(
        self,
        image_blob_name,
        bboxes="bboxes",
        labels="labels",
        scores="scores",
    ):
        try:
            parser = SingleOutputParser(self.outputs)
            self.logger.debug("\tUsing SSD model with single output parser")
            return parser
        except ValueError:
            pass

        try:
            parser = MultipleOutputParser(self.outputs, bboxes, scores, labels)
            self.logger.debug("\tUsing SSD model with multiple output parser")
            return parser
        except ValueError:
            pass

        try:
            parser = BoxesLabelsParser(self.outputs, (self.w, self.h))
            self.logger.debug('\tUsing SSD model with "boxes-labels" output parser')
            return parser
        except ValueError:
            pass
        msg = "Unsupported model outputs"
        raise ValueError(msg)

    def _parse_outputs(self, outputs):
        return self.output_parser(outputs)
