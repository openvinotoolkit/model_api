# Instance Segmentation

## Description

Instance segmentation model aims to detect and segment objects in an image. It is an extension of object detection, where each object is segmented into a separate mask. The model outputs a list of segmented objects, each containing a mask, bounding box, score and class label.

## OpenVINO Model Specifications

### Inputs

A single input image of shape (H, W, 3) where H and W are the height and width of the image, respectively.

### Outputs

Instance segmentation model outputs a list of segmented objects (i.e `list[SegmentedObject]`)wrapped in `InstanceSegmentationResult.segmentedObjects`, each containing the following attributes:

- `mask` (numpy.ndarray) - A binary mask of the object.
- `score` (float) - Confidence score of the object.
- `id` (int) - Class label of the object.
- `str_label` (str) - String label of the object.
- `xmin` (int) - X-coordinate of the top-left corner of the bounding box.
- `ymin` (int) - Y-coordinate of the top-left corner of the bounding box.
- `xmax` (int) - X-coordinate of the bottom-right corner of the bounding box.
- `ymax` (int) - Y-coordinate of the bottom-right corner of the bounding box.


## Example

```python
import cv2
from model_api.models import MaskRCNNModel

# Load the model
model = MaskRCNNModel.create_model("model.xml")

# Forward pass
predictions = model(image)

# Iterate over the segmented objects
for pred_obj in predictions.segmentedObjects:
    pred_mask = pred_obj.mask
    pred_score = pred_obj.score
    label_id = pred_obj.id
    bbox = [pred_obj.xmin, pred_obj.ymin, pred_obj.xmax, pred_obj.ymax]
```
