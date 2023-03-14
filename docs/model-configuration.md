# Model configuration
Model's static method `create_model()` configures a created model with values taken from `configuration` dict function argument and from model's intermediate representation (IR) stored in `.xml` in `model_info` section of `rt_info`. Values provided in `configuration` have priority over values in IR `rt_info`. If no value is specified in `configuration` nor in `rt_info` the default value for a model wrapper is used. For Python configuration values are accessible as model wrapper member fields.
## List of values
The list features only model wrappers which intoduce new configuration values in their hirachy.
1. `model_type`: str - name of a model wrapper to be created
### `ImageModel` and its subclasses
1. `mean_values`: List - Normalization values, which will be subtracted from image channels for image-input layer during preprocessing
1. `scale_values`: List - Normalization values, which will divide the image channels for image-input layer
1. `reverse_input_channels`: bool - Reverse the input channel order
1. `resize_type`: str - crop, standard, fit_to_window or fit_to_window_letterbox
1. `embed_preprocessing`: bool - Whether to embed preprocessing into the model
#### `ClassificationModel`
1. `topk`: int - number of most likely labels
1. `labels`: List - list of class labels
1. `path_to_labels`: str - path to file with labels. Overrides the labels, if they sets via 'labels' parameter
#### `DetectionModel` and its subclasses
1. `confidence_threshold`: float - probability threshold value for bounding box filtering
1. `labels`: List - List of class labels
1. `path_to_labels`: str - path to file with labels. Overrides the labels, if they sets via `labels` parameter
##### `CTPN`
1. `iou_threshold`: float - threshold for non-maximum suppression (NMS) intersection over union (IOU) filtering
1. `input_size`: List - image resolution which is going to be processed. Reshapes network to match a given size
##### `FaceBoxes`
1. `iou_threshold`: float - threshold for non-maximum suppression (NMS) intersection over union (IOU) filtering
##### `NanoDet`
1. `iou_threshold`: float - threshold for non-maximum suppression (NMS) intersection over union (IOU) filtering
1. `num_classes`: int - number of classes
##### `UltraLightweightFaceDetection`
1. `iou_threshold`: float - threshold for non-maximum suppression (NMS) intersection over union (IOU) filtering
##### `YOLO` and its subclasses
1. `iou_threshold`: float - threshold for non-maximum suppression (NMS) intersection over union (IOU) filtering
###### `YoloV4`
1. `anchors`: List - list of custom anchor values
1. `masks`: List - list of mask, applied to anchors for each output layer
###### `YOLOX`
1. `iou_threshold`: float - threshold for non-maximum suppression (NMS) intersection over union (IOU) filtering
#### `HpeAssociativeEmbedding`
1. `target_size`: int - image resolution which is going to be processed. Reshapes network to match a given size
1. `aspect_ratio`: float - image aspect ratio which is going to be processed. Reshapes network to match a given size
1. `confidence_threshold`: float - pose confidence threshold
1. `delta`: float
1. `size_divisor`: int - width and height of the rehaped model will be a multiple of this value
1. `padding_mode`: str - center or right_bottom
#### `OpenPose`
1. `target_size`: int - image resolution which is going to be processed. Reshapes network to match a given size
1. `aspect_ratio`: float - image aspect ratio which is going to be processed. Reshapes network to match a given size
1. `confidence_threshold`: float - pose confidence threshold
1. `upsample_ratio`: int - upsample ratio of a model backbone
1. `size_divisor`: int - width and height of the rehaped model will be a multiple of this value
#### `MaskRCNNModel`
1. `confidence_threshold`: float - probability threshold value for bounding box filtering
1. `labels`: List - list of class labels
1. `path_to_labels`: str - path to file with labels. Overrides the labels, if they sets via `labels` parameter
#### `SegmentationModel` and its subclasses
1. `labels`: List - list of class labels
1. `path_to_labels`: str - path to file with labels. Overrides the labels, if they sets via 'labels' parameter
### `Bert` and its subclasses
1. `vocab`: Dict - mapping from string token to int
1. `input_names`: str - comma-separated names of input layers
1. `enable_padding`: bool - should be input sequence padded to max sequence len or not
#### `BertQuestionAnswering`
1. `output_names`: str - comma-separated names of output layers
1. `max_answer_token_num`: int
1. `squad_ver`: str - SQuAD dataset version used for training. Affects how postprocessing

> Geti Anomaly model wrapper adds `image_threshold`, `pixel_threshold`, `min`, `max`, `threshold`.