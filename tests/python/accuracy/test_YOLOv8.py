import functools
import os
from pathlib import Path

import cv2
import numpy as np
import openvino.runtime as ov
import pytest
from openvino.model_api.models import YOLOv5
import ultralytics
import torch
import types


def _init_predictor(yolo):
    yolo.predict(np.empty([1, 1, 3], np.uint8))


@functools.lru_cache(maxsize=1)
def _cached_models(pt):
    export_dir = Path(
        ultralytics.YOLO(Path(os.environ["DATA"]) / "ultralytics" / pt, "detect").export(format="openvino", half=True)
    )
    impl_wrapper = YOLOv5.create_model(export_dir / (pt.stem + ".xml"), device="CPU")
    ref_wrapper = ultralytics.YOLO(export_dir, "detect")
    ref_wrapper.overrides["imgsz"] = (impl_wrapper.w, impl_wrapper.h)
    _init_predictor(ref_wrapper)
    ref_wrapper.predictor.model.ov_compiled_model = ov.Core().compile_model(
        ref_wrapper.predictor.model.ov_model, "CPU"
    )
    ref_dir = export_dir / "ref"
    ref_dir.mkdir(exist_ok=True)
    return impl_wrapper, ref_wrapper, ref_dir


def _impaths(all):
    """
    It's impossible to pass fixture as argument for
    @pytest.mark.parametrize, so it can't take a cmd arg. Use env var
    instead. Another solution was to define
    pytest_generate_tests(metafunc) in conftest.py
    """
    impaths = sorted(
        file
        for file in (Path(os.environ["DATA"]) / "coco128/images/train2017/").iterdir()
        if all or file.name
        not in {  # This images fail because image preprocessing is imbedded into the model
            "000000000143.jpg",
            "000000000491.jpg",
            "000000000536.jpg",
            "000000000581.jpg",
        }
    )
    if not impaths:
        raise RuntimeError(
            f"{Path(os.environ['DATA']) / 'coco128/images/train2017/'} is empty"
        )
    return impaths


@pytest.mark.parametrize("impath", _impaths(all=False))
@pytest.mark.parametrize("pt", [Path("yolov5mu.pt"), Path("yolov8l.pt")])
def test_accuracy_detector(impath, pt):
    impl_wrapper, ref_wrapper, ref_dir = _cached_models(pt)
    im = cv2.imread(str(impath))
    assert im is not None
    impl_preds = impl_wrapper(im)
    pred_boxes = np.array(
        [
            (
                impl_pred.xmin,
                impl_pred.ymin,
                impl_pred.xmax,
                impl_pred.ymax,
                impl_pred.score,
                impl_pred.id,
            )
            for impl_pred in impl_preds.objects
        ],
        dtype=np.float32,
    )
    ref_predictions = ref_wrapper.predict(im)
    assert 1 == len(ref_predictions)
    ref_boxes = ref_predictions[0].boxes.data.numpy()
    if 0 == pred_boxes.size == ref_boxes.size:
        return  # np.isclose() doesn't work for empty arrays
    ref_boxes[:, :4] = np.round(ref_boxes[:, :4], out=ref_boxes[:, :4])
    assert np.isclose(
        pred_boxes[:, :4], ref_boxes[:, :4], 0, 1
    ).all()  # Allow one pixel deviation because image preprocessing is imbedded into the model
    assert np.isclose(pred_boxes[:, 4], ref_boxes[:, 4], 0.0, 0.02).all()
    assert (pred_boxes[:, 5] == ref_boxes[:, 5]).all()
    with open(ref_dir / impath.with_suffix(".txt").name, "w") as file:
        print(impl_preds, end="", file=file)


class Metrics(ultralytics.models.yolo.detect.DetectionValidator):
    @torch.inference_mode()
    def evaluate(self, yolo, names, dataset_yml):
        # TODO: multilabel, both scales, ceil instead of round
        self.data = ultralytics.data.utils.check_det_dataset(dataset_yml)
        dataloader = self.get_dataloader(self.data[self.args.split], batch_size=1)
        dataloader.dataset.transforms.transforms = (lambda di: {
            'batch_idx': torch.zeros(len(di['instances'])),
            'bboxes': torch.from_numpy(di['instances'].bboxes),
            'cls': torch.from_numpy(di['cls']),
            'img': torch.empty(1, 1, 1),
            'im_file': di['im_file'],
            'ori_shape': di['ori_shape'],
            'ratio_pad': [(1.0, 1.0), (0, 0)],
        },)
        self.init_metrics(names)
        for batch in dataloader:
            im = cv2.imread(batch['im_file'][0])
            pred = torch.tensor(
                [[
                    (
                        impl_pred.xmin / im.shape[1],
                        impl_pred.ymin / im.shape[0],
                        impl_pred.xmax / im.shape[1],
                        impl_pred.ymax / im.shape[0],
                        impl_pred.score,
                        impl_pred.id,
                    )
                    for impl_pred in yolo(im).objects
                ]],
                dtype=torch.float32,
            )
            if not pred.numel():
                pred = torch.empty(1, 0, 6)
            # pred = yolo.predict(im, conf=0.001)[0].boxes.data
            # pred[:, (0, 2)] /= im.shape[1]
            # pred[:, (1, 3)] /= im.shape[0]
            self.update_metrics(pred, batch)
        return self.get_stats()


@pytest.mark.parametrize("pt", [Path("yolov8n.pt")])#[Path("yolov8n.pt")])#, Path("yolov5n6u.pt")])
def test_detector_metric(pt):
    export_dir = export_dir = Path(
        ultralytics.YOLO(Path(os.environ["DATA"]) / "ultralytics" / pt, "detect").export(format="openvino", half=True)
    )
    impl_wrapper = YOLOv5.create_model(export_dir / pt.with_suffix(".xml"), device="CPU", configuration={"confidence_threshold": 0.001})

    kek = ultralytics.YOLO(export_dir, "detect")
    kek.overrides = {"imgsz": (impl_wrapper.w, impl_wrapper.h)}  # TODO: wh or hw?

    mAP50_95 = Metrics().evaluate(impl_wrapper, types.SimpleNamespace(names={idx: label for idx, label in enumerate(impl_wrapper.labels)}), "coco8.yaml")["metrics/mAP50-95(B)"]
    print(f"{mAP50_95:.99f}")
    assert mAP50_95 >= 0.609230627349896747269042407424421980977058410644531250000000000000000000000000000000000000000000000
    # pytorch ceil: 0.606196548258608802761671086045680567622184753417968750000000000000000000000000000000000000000000000
    # vino ceil: 0.609230627349896747269042407424421980977058410644531250000000000000000000000000000000000000000000000
    # pytorch no ceil: 0.606196548258608802761671086045680567622184753417968750000000000000000000000000000000000000000000000
    # pytorch no ceil: 0.606196548258608802761671086045680567622184753417968750000000000000000000000000000000000000000000000

    mAP50_95 = Metrics().evaluate(impl_wrapper, types.SimpleNamespace(names={idx: label for idx, label in enumerate(impl_wrapper.labels)}), "coco128.yaml")["metrics/mAP50-95(B)"]
    print(f"{mAP50_95:.99f}")
    assert mAP50_95 >= 0.439413760605130543357432770790182985365390777587890625000000000000000000000000000000000000000000000
    # vino no ceil: 0.454416461188190345943382908444618806242942810058593750000000000000000000000000000000000000000000000
    # vino ceil: 0.453295803020367371605203743456513620913028717041015625000000000000000000000000000000000000000000000
    # pytorch ceil: 0.456389257039404527827031188280670903623104095458984375000000000000000000000000000000000000000000000
    # pytorch no ceil: 0.456607101106700774550972710130736231803894042968750000000000000000000000000000000000000000000000000
    # pytorch no ceil: 0.456607101106700774550972710130736231803894042968750000000000000000000000000000000000000000000000000

    mAP50_95 = Metrics().evaluate(impl_wrapper, types.SimpleNamespace(names={idx: label for idx, label in enumerate(impl_wrapper.labels)}), "coco.yaml")["metrics/mAP50-95(B)"]
    print(f"{mAP50_95:.99f}")
    assert mAP50_95 >= 0.365912774507280713631729440749040804803371429443359375000000000000000000000000000000000000000000000
    # vino ceil: 0.371225813018740136151052411150885745882987976074218750000000000000000000000000000000000000000000000
    # veno no ceil: 0.371190303780130126387604150295373983681201934814453125000000000000000000000000000000000000000000000
    # pytorch ceil: 0.370275365583863536045328146428801119327545166015625000000000000000000000000000000000000000000000000
    # pytorch no ceil: 0.370294050861252221906738668621983379125595092773437500000000000000000000000000000000000000000000000
    # pytorch no ceil: 0.370294050861252221906738668621983379125595092773437500000000000000000000000000000000000000000000000


    # mAP50_95 = Metrics().evaluate(kek, types.SimpleNamespace(names={idx: label for idx, label in enumerate(impl_wrapper.labels)}), "coco8.yaml")["metrics/mAP50-95(B)"]
    # print(f"{mAP50_95:.99f}")
    # assert mAP50_95 >= 0.609230627349896747269042407424421980977058410644531250000000000000000000000000000000000000000000000

    # mAP50_95 = Metrics().evaluate(kek, types.SimpleNamespace(names={idx: label for idx, label in enumerate(impl_wrapper.labels)}), "coco128.yaml")["metrics/mAP50-95(B)"]
    # print(f"{mAP50_95:.99f}")
    # assert mAP50_95 >= 0.453236284183963278326956469754804857075214385986328125000000000000000000000000000000000000000000000
    # # 0.454416461188190345943382908444618806242942810058593750000000000000000000000000000000000000000000000 final
    # # 0.454480642834196590928996783986804075539112091064453125000000000000000000000000000000000000000000000
    # # 0.453149872092887873176181301460019312798976898193359375000000000000000000000000000000000000000000000
    # # AUTO: 0.450826465645132734572086974367266520857810974121093750000000000000000000000000000000000000000000000
    # mAP50_95 = Metrics().evaluate(kek, types.SimpleNamespace(names={idx: label for idx, label in enumerate(impl_wrapper.labels)}), "coco.yaml")["metrics/mAP50-95(B)"]
    # print(f"{mAP50_95:.99f}")
    # assert mAP50_95 >= 0.370558668805542223978477522905450314283370971679687500000000000000000000000000000000000000000000000
    # 0.371198321535742781218658592479187063872814178466796875000000000000000000000000000000000000000000000
    # 0.371198321535742781218658592479187063872814178466796875000000000000000000000000000000000000000000000
    # 0.371190303780130126387604150295373983681201934814453125000000000000000000000000000000000000000000000 final

    # mAP50_95 = yolo.detect.DetectionValidator(args={"data": "coco128.yaml", "imgsz": impl_wrapper.w})(kek, asdf(), model=kek.model)["metrics/mAP50-95(B)"]
    # print(f"{mAP50_95:.99f}")
    # assert mAP50_95 >= 0.450826465645132734572086974367266520857810974121093750000000000000000000000000000000000000000000000
    # # 0.451171826006411980092281055476632900536060333251953125000000000000000000000000000000000000000000000
    # # 0.451189916976566962603101273998618125915527343750000000000000000000000000000000000000000000000000000
    # # CPU
    # # 0.453149872092887873176181301460019312798976898193359375000000000000000000000000000000000000000000000

    # mAP50_95 = yolo.detect.DetectionValidator(args={"data": "coco.yaml", "imgsz": impl_wrapper.w})(kek, asdf(), model=kek.model)["metrics/mAP50-95(B)"]
    # print(f"{mAP50_95:.99f}")
    # assert mAP50_95 >= 0.369873965487229283688463965518167242407798767089843750000000000000000000000000000000000000000000000
    # # 0.370175703174132286754627330083167180418968200683593750000000000000000000000000000000000000000000000
    # # CPU
    # # 0.370345057538419675235985550898476503789424896240234375000000000000000000000000000000000000000000000

    # mAP50_95 = kek.val(data="coco.yaml").box.map
    # print(f"{mAP50_95:.99f}")

    # model = YOLO("/home/wov/Downloads/yolov8n.pt", "detect")
    # metrics = model.val(data="coco.yaml")
    # yolo = YOLOv5.create_model(export_dir / (pt.stem + ".xml"), configuration={"pad_value": 10, "resize_type": "asdf"}, device="CPU")

    # print(ultralytics.models.yolo.detect.DetectionValidator(args={"data": "coco.yaml", "device": "cpu"})(model="yolov8n.pt")["metrics/mAP50-95(B)"])
