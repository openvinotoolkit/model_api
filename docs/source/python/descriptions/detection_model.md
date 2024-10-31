# Detection Model

## Description

Detection model aims to detect objects in an image. The model outputs a list of detected objects, each containing a bounding box, score and class label.

## OpenVINO Model Specifications

### Inputs

A single input image of shape (H, W, 3) where H and W are the height and width of the image, respectively.

### Outputs

Detection model outputs a list of detection objects (i.e `list[Detection]`) wrapped in `DetectionResult`, each object containing the following attributes:

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
from model_api.models import SSD

# Load the model
model = SSD.create_model("model.xml")

# Forward pass
predictions = model(image)

# Iterate over the segmented objects
for pred_obj in predictions.objects:
    pred_score = pred_obj.score
    label_id = pred_obj.id
    bbox = [pred_obj.xmin, pred_obj.ymin, pred_obj.xmax, pred_obj.ymax]
```
