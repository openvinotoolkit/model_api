# OpenVINO Model API
Model API is a set of wrapper classes for particular tasks and model architectures, simplifying data preprocess and postprocess as well as routine procedures (model loading, asynchronous execution, etc.). It is aimed at simplifying end-to-end model inference for different deployment scenarious, including local execution and serving. The Model API is based on the OpenVINO inference API.

## How it works
Model API searches for additional information required for model inference, data, pre/postprocessing, label names, etc. directly in OpenVINO Intermediate Representation. This information is used to prepare the inference data, process and ouput the inference results in a human-readable format.

## Features
- Python and C++ API
- Automatic prefetch of public models from [OpenVINO Model Zoo](https://github.com/openvinotoolkit/open_model_zoo) (Python only)
- Syncronous and asynchronous inference
- Local inference and servring through the rest API (Python only)
- Model preprocessing embedding for faster inference

## Installation
### Python
- Clone this repository
- Navigate to `model_api/python` folder
- Run `pip install .`
### C++
- Install dependencies. For installation on Ubuntu, you can use the following script:
  ```bash
  chmod +x model_api/cpp/install_dependencies.sh
  sudo model_api/cpp/install_dependencies.sh
  ```

- Build library:
   - Create `build` folder and navigate into it:
   ```
   mkdir build && cd build
   ```
   - Run cmake:
   ```
   cmake ../model_api/cpp -DOpenCV_DIR=<OpenCV cmake dir> -DOpenVINO_DIR=<OpenVINO cmake dir>
   ```
   - Build:
   ```
   cmake --build . -j
   ```
   - To build a `.tar.gz` package with the library, run:
   ```
    make package
    ```

## Usage
### Python
```python
from openvino.model_api.models import DetectionModel

# Create a model (downloaded and cached automatically for OpenVINO Model Zoo models)
# Use URL to work with served model, e.g. "localhost:9000/models/ssd300"
ssd = DetectionModel.create_model("ssd300")

# Run synchronous inference locally
detections = ssd(image)  # image is numpy.ndarray

# Print the list of Detection objects with box coordinates, confidence and label string
print(f"Detection results: {detections}")
```

### C++
```cpp
#include <models/detection_model.h>
#include <models/results.h>

// Load the model fetched using Python API
auto model = DetectionModel::create_model("~/.cache/omz/public/ssd300/FP16/ssd300.xml");

// Run synchronous inference locally
auto result = model->infer(image); // image is cv::Mat

// Iterate over the vector of DetectedObject with box coordinates, confidence and label string
for (auto& obj : result->objects) {
    std::cout << obj.label << " | " << obj.confidence << " | " << int(obj.x) << " | " << int(obj.y) << " | "
        << int(obj.x + obj.width) << " | " << int(obj.y + obj.height) << std::endl;
}
```

For more details please refer to the [examples](https://github.com/openvinotoolkit/model_api/tree/master/examples) of this project.

## Supported models
### Python:
- Image Classification:
  - [OpenVINO Model Zoo models](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/index.md#classification-models)
- Object Detection:
  - [OpenVINO Model Zoo models](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/index.md#object-detection-models):
    - SSD-based models (e.g. "ssd300", "ssdlite_mobilenet_v2", etc.)
    - YOLO-based models (e.g. "yolov3", "yolov4", etc.)
    - CTPN: "ctpn"
    - DETR: "detr-resnet50"
    - CenterNet: "ctdet_coco_dlav0_512"
    - FaceBoxes: "faceboxes-pytorch"
    - RetinaFace: "retinaface-resnet50-pytorch"
    - Ultra Lightweight Face Detection: "ultra-lightweight-face-detection-rfb-320" and "ultra-lightweight-face-detection-slim-320"
    - NanoDet with ShuffleNetV2: "nanodet-m-1.5x-416"
    - NanoDet Plus with ShuffleNetV2: "nanodet-plus-m-1.5x-416"
- Semantic Segmentation:
  - [OpenVINO Model Zoo models](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/index.md#semantic-segmentation-models)
- Instance Segmentation:
  - [OpenVINO Model Zoo models](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/index.md#instance-segmentation-models)


### C++:
- Image Classification:
  - [OpenVINO Model Zoo models](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/index.md#classification-models)
- Object Detection:
  - SSD-based models (e.g. "ssd300", "ssdlite_mobilenet_v2", etc.)
    - YOLO-based models (e.g. "yolov3", "yolov4", etc.)
    - CenterNet: "ctdet_coco_dlav0_512"
    - FaceBoxes: "faceboxes-pytorch"
    - RetinaFace: "retinaface-resnet50-pytorch"
- Semantic Segmentation:
  - [OpenVINO Model Zoo models](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/index.md#semantic-segmentation-models)

[Model configuration](docs/model-configuration.md) discusses different options of model creation and possible configurations.
