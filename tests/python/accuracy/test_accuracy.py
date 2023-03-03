import math
import os
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path
import json

import cv2
import numpy as np
import pytest
import requests


from openvino.model_api.models import (
    ClassificationModel,
    DetectionModel,
    SegmentationModel,
)


def process_output(ouput, model_type):
    if model_type == DetectionModel.__name__:
        return f"{ouput}"
    elif model_type == ClassificationModel.__name__:
        return f"({ouput[0]}, {ouput[1]}, {ouput[2]:.3f})"
    else:
        raise ValueError("Unknown model type to precess ouput")

def read_config(path: Path):
    with open(path, "r") as f:
        return json.load(f)

@pytest.fixture(scope="session")
def data(pytestconfig):
    return pytestconfig.getoption("data")

@pytest.fixture(scope="session")
def dump(pytestconfig):
    return pytestconfig.getoption("dump")

@pytest.fixture(scope="session")
def result(pytestconfig):
    return pytestconfig.test_results

@pytest.mark.parametrize(
    ("model_data"), read_config(Path(__file__).resolve().parent / "public_scope.json")
)
def test_image_models(data, dump, result, model_data):
    model = eval(model_data["type"]).create_model(model_data["name"], download_dir=data)
    
    test_result = []
    
    if dump:
        result.append(model_data)
        inference_results = []

    for test_data in model_data["test_data"]:
        image_path = Path(data) / "coco128/images/train2017/" / test_data["image"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError("Failed to read the image")
        outputs = model(image)

        image_result = []
        
        for i, output in enumerate(outputs):
            output_str = process_output(output, model_data["type"])
            test_result.append(test_data["reference"][i] == output_str)
            
            if dump:
                image_result.append(output_str)
                
        if dump:
            inference_results.append({"image": test_data["image"], "reference": image_result})
    if dump:      
        result[-1]["test_data"] = inference_results
        
    assert all(test_result)

