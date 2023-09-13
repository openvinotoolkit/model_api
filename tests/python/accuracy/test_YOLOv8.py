import functools
import os
from distutils.dir_util import copy_tree
from pathlib import Path

import cv2
import numpy as np
import openvino.runtime as ov
import pytest
import torch
import torchvision.transforms as T
from openvino.model_api.models import YOLOv5
from ultralytics import YOLO
from ultralytics.yolo.engine.results import Results


class CenterCrop:
    # YOLOv8 CenterCrop class for image preprocessing, i.e. T.Compose([CenterCrop(size), ToTensor()])
    def __init__(self, size=640):
        """Converts an image from numpy array to PyTorch tensor."""
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size

    def __call__(self, im):  # im = np.array HWC
        imh, imw = im.shape[:2]
        m = min(imh, imw)  # min dimension
        top, left = (imh - m) // 2, (imw - m) // 2
        return cv2.resize(
            im[top : top + m, left : left + m],
            (self.w, self.h),
            interpolation=cv2.INTER_LINEAR,
        )


class ToTensor:
    # YOLOv8 ToTensor class for image preprocessing, i.e. T.Compose([LetterBox(size), ToTensor()])
    def __init__(self, half=False):
        """Initialize YOLOv8 ToTensor object with optional half-precision support."""
        super().__init__()
        self.half = half

    def __call__(self, im):  # im = np.array HWC in BGR order
        im = np.ascontiguousarray(
            im.transpose((2, 0, 1))[::-1]
        )  # HWC to CHW -> BGR to RGB -> contiguous
        im = torch.from_numpy(im)  # to torch
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0-255 to 0.0-1.0
        return im


@pytest.fixture(scope="session")
def data(pytestconfig):
    return Path(pytestconfig.getoption("data"))


def _init_predictor(yolo):
    yolo.predict(np.empty([1, 1, 3], np.uint8))


@functools.lru_cache(maxsize=1)
def _cached_models(folder, pt):
    pt = Path(pt)
    export_dir = Path(
        YOLO(folder / "ultralytics/detectors" / pt, "detect").export(format="openvino")
    )
    impl_wrapper = YOLOv5.create_model(export_dir / (pt.stem + ".xml"), device="CPU")
    ref_wrapper = YOLO(export_dir, "detect")
    ref_wrapper.overrides["imgsz"] = (impl_wrapper.w, impl_wrapper.h)
    _init_predictor(ref_wrapper)
    ref_wrapper.predictor.model.ov_compiled_model = ov.Core().compile_model(
        ref_wrapper.predictor.model.ov_model, "CPU"
    )
    ref_dir = export_dir / "ref"
    ref_dir.mkdir(exist_ok=True)
    return impl_wrapper, ref_wrapper, ref_dir


def test_detector(impath, data, pt):
    impl_wrapper, ref_wrapper, ref_dir = _cached_models(data, pt)
    im = cv2.imread(str(impath))
    assert im is not None
    impl_preds = impl_wrapper(im)
    pred_boxes = np.array(
        [
            [
                impl_pred.xmin,
                impl_pred.ymin,
                impl_pred.xmax,
                impl_pred.ymax,
                impl_pred.score,
                impl_pred.id,
            ]
            for impl_pred in impl_preds.objects
        ],
        dtype=np.float32,
    )
    ref_predictions = ref_wrapper.predict(im)
    assert 1 == len(ref_predictions)
    ref_boxes = ref_predictions[0].boxes.data.numpy()
    with open(ref_dir / impath.with_suffix(".txt").name, "w") as file:
        print(impl_preds, file=file)
    if 0 == pred_boxes.size == ref_boxes.size:
        return  # np.isclose() doesn't work for empty arrays
    ref_boxes[:, :4] = np.round(ref_boxes[:, :4], out=ref_boxes[:, :4])
    assert np.isclose(
        pred_boxes[:, :4], ref_boxes[:, :4], 0, 1
    ).all()  # allow one pixel deviation because image preprocessing is imbedded into the model
    assert np.isclose(pred_boxes[:, 4], ref_boxes[:, 4], 0.0, 0.02).all()
    assert (pred_boxes[:, 5] == ref_boxes[:, 5]).all()


def test_classifier(data):
    # export_path = YOLO("https://github.com/ultralytics/assets/releases/download/v0.0.0/YOLOv8n-cls.pt").export(format="openvino")
    export_path = YOLO(
        "/home/wov/r/ultralytics/examples/YOLOv8-CPP-Inference/build/YOLOv8n-cls.pt"
    ).export(format="openvino")
    xmls = [file for file in os.listdir(export_path) if file.endswith(".xml")]
    if 1 != len(xmls):
        raise RuntimeError(f"{export_path} must contain one .xml file")
    ref_wrapper = YOLO(export_path)
    ref_wrapper.overrides["imgsz"] = 224
    im = cv2.imread(data + "/coco128/images/train2017/000000000074.jpg")
    ref_predictions = ref_wrapper.predict(im)

    model = ov.Core().compile_model(f"{export_path}/{xmls[0]}")
    orig_imgs = [im]

    transforms = T.Compose([CenterCrop(224), ToTensor()])

    img = torch.stack([transforms(im) for im in orig_imgs], dim=0)
    img = img if isinstance(img, torch.Tensor) else torch.from_numpy(img)
    img.float()  # uint8 to fp16/32

    preds = next(iter(model({0: img}).values()))
    preds = torch.from_numpy(preds)

    results = []
    for i, pred in enumerate(preds):
        orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
        results.append(
            Results(
                orig_img=orig_img,
                path=None,
                names=ref_wrapper.predictor.model.names,
                probs=pred,
            )
        )

    for i in range(len(results)):
        assert result.boxes == ref_predictions.boxes
        assert result.keypoints == ref_predictions.keypoints
        assert result.keys == ref_predictions.keys
        assert result.masks == ref_predictions.masks
        assert result.names == ref_predictions.names
        assert (result.orig_img == ref_predictions.orig_img).all()
        assert (result.probs == ref_predictions.probs).all()
