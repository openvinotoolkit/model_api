# OpenVINO Model API
Model API is a set of wrapper classes for particular tasks and model architectures, simplifying data preprocess and postprocess as well as routine procedures (model loading, asynchronous execution, etc.). It is aimed at simplifying end-to-end model inference for different deployment scenarious, including local execution and serving. The Model API is based on the OpenVINO inference API.

## Features
- Python and C++ API
- Automatic prefetch of public models from [OpenVINO Model Zoo](https://github.com/openvinotoolkit/open_model_zoo) (Python only)
- Syncronous and asynchronous inference
- Local inference and servring through the rest API (Python only).
- Model preprocessing embedding for faster inference (Python only)

## Installation
### Python
- Clone this repository
- Navigate to `model_api/python` folder
- Run `pip install .`
### C++
- Install dependencies. For installation on Ubuntu, you can use the following script:
  ```bash
  sudo model_api/install_dependencies.sh
  ```

- Build library:
   - Create `build` folder and navigate into it:
   ```
   mkdir build && cd build
   ```
   - Run cmake:
   ```
   cmake ../model_api/cpp
   ```
   - Build:
   ```
   make -j 
   ```
   - To build a `.tar.gz` package with the library, run:
   ```
    make package
    ```

## Usage
For more details please refer to the [examples](https://github.com/openvinotoolkit/model_api/tree/master/examples) of this project.
### Local inference
- Python
```python
from openvino.model_api.models import DetectionModel

# Create a model (downloaded and cached automatically)
ssd = DetectionModel.create_model("ssd300")

# Run synchronous inference locally
detections = ssd(image) # list of Detection objects with box coordinates, confidence and label string
```
- C++
```cpp
#include <models/detection_model.h>
#include <models/input_data.h>
#include <models/results.h>

// Load the model fetched using Python API
auto model = DetectionModel::create_model("~/.cache/omz/public/ssd300/FP16/ssd300.xml");

// Run synchronous inference locally
 auto result = model->infer(ImageInputData(image));
 
 // Access and process results
 auto detections = result->asRef<DetectionResult>();
 for (auto& obj : result.objects) {
 ...
 }
```

More examples of how to use asynchronous inference and serving are coming soon.