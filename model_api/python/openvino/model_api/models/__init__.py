"""
 Copyright (C) 2021-2023 Intel Corporation

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


from .background_matting import (
    ImageMattingWithBackground,
    PortraitBackgroundMatting,
    VideoBackgroundMattingModel,
)
from .bert import BertEmbeddingModel, BertNamedEntityRecognitionModel, BertQuestionAnsweringModel
from .centernet import CenterNetModel
from .classification import ClassificationModel
from .ctpn import CTPNModel
from .deblurring import DeblurringModel
from .detection_model import DetectionModel
from .detr import DETRModel
from .faceboxes import FaceBoxesModel
from .hpe_associative_embedding import HpeAssociativeEmbeddingModel
from .image_model import ImageModel
from .instance_segmentation import MaskRCNNModel, YolactModel
from .model import Model
from .monodepth import MonoDepthModel
from .nanodet import NanoDetModel, NanoDetPlusModel
from .open_pose import OpenPoseModel
from .retinaface import RetinaFaceModel, RetinaFacePyTorchModel
from .segmentation import SalientObjectDetectionModel, SegmentationModel
from .ssd import SSDModel
from .ultra_lightweight_face_detection import UltraLightweightFaceDetectionModel
from .utils import (
    RESIZE_TYPES,
    Detection,
    DetectionWithLandmarks,
    InputTransform,
    OutputTransform,
)
from .yolo import YOLOModel, YOLOFModel, YOLOXModel, YoloV3ONNXModel, YoloV4Model

classification_models = [
    "resnet-18-pytorch",
    "se-resnext-50",
    "mobilenet-v3-large-1.0-224-tf",
    "efficientnet-b0-pytorch",
]

detection_models = [
    # "face-detection-retail-0044", # resize_type is wrong or missed in model.yml
    "yolo-v4-tf",
    "ssd_mobilenet_v1_fpn_coco",
    "ssdlite_mobilenet_v2",
]

segmentation_models = [
    "fastseg-small",
]


__all__ = [
    "BertEmbeddingModel",
    "BertNamedEntityRecognitionModel",
    "BertQuestionAnsweringModel",
    "CenterNetModel",
    "ClassificationModel",
    "CTPNModel",
    "DeblurringModel",
    "DetectionModel",
    "DetectionWithLandmarks",
    "DETRModel",
    "FaceBoxesModel",
    "HpeAssociativeEmbeddingModel",
    "ImageMattingWithBackground",
    "ImageModel",
    "InputTransform",
    "MaskRCNNModel",
    "Model",
    "MonoDepthModel",
    "NanoDetModel",
    "NanoDetPlusModel",
    "OpenPoseModel",
    "OutputTransform",
    "PortraitBackgroundMatting",
    "RESIZE_TYPES",
    "RetinaFaceModel",
    "RetinaFacePyTorchModel",
    "SalientObjectDetectionModel",
    "SegmentationModel",
    "SSDModel",
    "UltraLightweightFaceDetectionModel",
    "VideoBackgroundMattingModel",
    "YolactModel",
    "YOLOModel",
    "YoloV3ONNXModel",
    "YoloV4Model",
    "YOLOFModel",
    "YOLOXModel",
    "Detection",
    "DetectionWithLandmarks",
    "classification_models" "detection_models",
    "segmentation_models",
]
