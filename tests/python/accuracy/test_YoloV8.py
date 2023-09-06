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
from openvino.model_api.models.utils import resize_image_letterbox
from ultralytics import YOLO
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import ops


# TODO: update docs
def patch_export(yolo):
    # TODO: move to https://github.com/ultralytics/ultralytics/
    if yolo.predictor is None:
        yolo.predict(
            np.empty([1, 1, 3], np.uint8)
        )  # YOLO.predictor is initialized by predict
    export_dir = Path(yolo.export(format="openvino"))
    xml = [path for path in export_dir.iterdir() if path.suffix == ".xml"]
    if 1 != len(xml):
        raise RuntimeError(f"{export_dir} must contain one .xml file")
    xml = xml[0]
    model = ov.Core().read_model(xml)
    tempxml = export_dir / "temp/temp.xml"
    ov.serialize(model, tempxml)
    del model
    binpath = xml.with_suffix(".bin")
    xml.unlink(missing_ok=True)
    binpath.unlink(missing_ok=True)
    tempxml.rename(xml)
    tempxml.with_suffix(".bin").rename(binpath)
    return export_dir


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


class LetterBox:
    """Resize image and padding for detection, instance segmentation, pose."""

    def __init__(
        self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, stride=32
    ):
        """Initialize LetterBox object with specific parameters."""
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride

    def __call__(self, labels=None, image=None):
        """Return updated labels and image with added border."""
        if labels is None:
            labels = {}
        img = labels.get("img") if image is None else image
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = (
                new_shape[1] / shape[1],
                new_shape[0] / shape[0],
            )  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2
        if labels.get("ratio_pad"):
            labels["ratio_pad"] = (labels["ratio_pad"], (dw, dh))  # for evaluation

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # add border

        if len(labels):
            labels = self._update_labels(labels, ratio, dw, dh)
            labels["img"] = img
            labels["resized_shape"] = new_shape
            return labels
        else:
            return img

    def _update_labels(self, labels, ratio, padw, padh):
        """Update labels."""
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(*labels["img"].shape[:2][::-1])
        labels["instances"].scale(*ratio)
        labels["instances"].add_padding(padw, padh)
        return labels


@pytest.fixture(scope="session")
def data(pytestconfig):
    return Path(pytestconfig.getoption("data"))


@functools.lru_cache(maxsize=1)
def cached_models(folder, pt):
    pt = Path(pt)
    yolo_folder = folder / "YOLOv8"
    yolo_folder.mkdir(exist_ok=True)  # TODO: maybe remove
    export_dir = patch_export(
        YOLO(yolo_folder / pt)
    )  # If there is no file it is downloaded
    copy_path = folder / "YOLOv8/detector" / pt.stem
    copy_tree(str(export_dir), str(copy_path))  # C++ tests expect a model here
    xml = copy_path / (pt.stem + ".xml")
    ref_dir = copy_path / "ref"
    ref_dir.mkdir(exist_ok=True)
    impl_wrapper = YOLOv5.create_model(
        xml, device="CPU", model_type="YOLOv5"
    )  # TODO: YOLOv5 vs v8
    ref_wrapper = YOLO(export_dir)
    ref_wrapper.overrides["imgsz"] = (impl_wrapper.w, impl_wrapper.h)
    if ref_wrapper.predictor is None:
        ref_wrapper.predict(
            np.empty([1, 1, 3], np.uint8)
        )  # YOLO.predictor is initialized by predict
    core = ov.Core()
    ref_wrapper.predictor.model.ov_compiled_model = core.compile_model(
        ref_wrapper.predictor.model.ov_model, "CPU"
    )
    compiled_model = core.compile_model(xml, "CPU")
    return impl_wrapper, ref_wrapper, compiled_model


# TODO: test save-load
def test_detector(impath, data, pt):
    impl_wrapper, ref_wrapper, compiled_model = cached_models(data, pt)
    im = cv2.imread(str(impath))
    if im is None:
        raise RuntimeError("Failed to read the image")
    impl_prediction = impl_wrapper(im)
    # with open(ref_dir / impath.with_suffix(".txt").name, "w") as file:
    #     for pred in impl_prediction:
    #         print(pred, file=file)
    ref_predictions = ref_wrapper.predict(im)
    assert 1 == len(ref_predictions)
    ref_predictions = ref_predictions[0]

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
            for impl_pred in impl_prediction
        ],
        dtype=np.float32,
    )
    ref_preprocessed = ref_wrapper.predictor.preprocess([im]).numpy()

    processed = resize_image_letterbox(
        im, (impl_wrapper.w, impl_wrapper.h), cv2.INTER_LINEAR, 114
    )
    processed = (
        processed[None][..., ::-1].transpose((0, 3, 1, 2)).astype(np.float32) / 255.0
    )
    assert (processed == ref_preprocessed).all()
    preds = next(iter(compiled_model({0: processed}).values()))
    preds = torch.from_numpy(preds)
    preds = ops.non_max_suppression(
        preds,
        ref_wrapper.predictor.args.conf,
        ref_wrapper.predictor.args.iou,
        agnostic=ref_wrapper.predictor.args.agnostic_nms,
        max_det=ref_wrapper.predictor.args.max_det,
        classes=ref_wrapper.predictor.args.classes,
    )
    pred = preds[0]
    pred[:, :4] = ops.scale_boxes(processed.shape[2:], pred[:, :4], im.shape)
    result = Results(
        orig_img=im, path=None, names=ref_wrapper.predictor.model.names, boxes=pred
    )

    # if impl_prediction.size:
    #     print((impl_prediction - preds[0].numpy()).max())
    #     assert np.isclose(impl_prediction, preds[0], 3e-3, 0.0).all()
    ref_boxes = ref_predictions.boxes.data.numpy().copy()
    if 0 == pred_boxes.size == ref_boxes.size:
        return  # np.isclose() doesn't work for empty arrays
    ref_boxes[:, :4] = np.round(ref_boxes[:, :4], out=ref_boxes[:, :4])
    assert np.isclose(
        pred_boxes[:, :4], ref_boxes[:, :4], 0, 1
    ).all()  # allow one pixel deviation because image resize is imbedded into the model
    assert np.isclose(
        pred_boxes[:, 4], ref_boxes[:, 4], 0.0, 0.02
    ).all()  # TODO: maybe stronger
    assert (pred_boxes[:, 5] == ref_boxes[:, 5]).all()
    assert (result.boxes.data == ref_predictions.boxes.data).all()
    assert result.boxes.orig_shape == ref_predictions.boxes.orig_shape
    assert result.keypoints == ref_predictions.keypoints
    assert result._keys == ref_predictions._keys
    assert result.masks == ref_predictions.masks
    assert result.names == ref_predictions.names
    assert (result.orig_img == ref_predictions.orig_img).all()
    assert result.probs == ref_predictions.probs


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
