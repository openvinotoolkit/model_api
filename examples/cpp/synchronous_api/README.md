# Synchronous API example
This example demonstrates how to use a C++ API of OpenVINO Model API for synchronous inference and its basic steps:
- Instantiate a model
- Model inference
- Process results

## Prerequisites
- Install third party dependencies by running the following script:
    ```bash
    chmod +x ../../../model_api/cpp/install_dependencies.sh
    sudo ../../../model_api/cpp/install_dependencies.sh
    ```
- Build example:
   - Create `build` folder and navigate into it:
   ```
   mkdir build && cd build
   ```
   - Run cmake:
   ```
   cmake ../
   ```
   - Build:
   ```
   make -j 
   ```
- Fetch a model by running a Python code with Model API, see Python [exaple](../../python/synchronous_api/README.md):
    ```python
    from openvino.model_api.models import DetectionModel

    model = DetectionModel.create_model("ssd_mobilenet_v1_fpn_coco",
                                    download_dir="tmp")
    ```

## Run example
To run the example, please execute the following command:
```bash
./synchronous_api ./tmp/public/ssd_mobilenet_v1_fpn_coco/FP16/ssd_mobilenet_v1_fpn_coco.xml <path_to_image>
```
