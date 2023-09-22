import functools
import os
from pathlib import Path

import cv2
import numpy as np
import openvino.runtime as ov
import pytest
from openvino.model_api.models import YOLOv5
from ultralytics import YOLO
from ultralytics.models import yolo



def _init_predictor(yolo):
    yolo.predict(np.empty([1, 1, 3], np.uint8))


@functools.lru_cache(maxsize=1)
def _cached_models(pt):
    export_dir = Path(
        YOLO(Path(os.environ["DATA"]) / "ultralytics" / pt, "detect").export(format="openvino", half=True)
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


@functools.lru_cache(maxsize=1)
def _cached_config_models(pt):
    export_dir = Path(
        YOLO(Path(os.environ["DATA"]) / "ultralytics" / pt, "detect").export(format="openvino", half=True)
    )
    return export_dir

from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.utils import LOGGER, TQDM, callbacks, colorstr, emojis

class ModelAPIValidator(yolo.detect.DetectionValidator):
    def __init__(self, save_dir, args):
        super().__init__(save_dir=save_dir, args=args)

    def __call__(self, model):
        """
        Supports validation of a pre-trained model if passed or a model being trained if trainer is passed (trainer
        gets priority).
        """
        self.training = False
        augment = self.args.augment and (not self.training)

        # self.device = model.device  # update device
        self.args.half = True   # TODO: maybe remove
        self.args.batch = 1  # TODO: maybe remove

        if isinstance(self.args.data, str) and self.args.data.split('.')[-1] in ('yaml', 'yml'):
            self.data = check_det_dataset(self.args.data)
        elif self.args.task == 'classify':
            self.data = check_cls_dataset(self.args.data, split=self.args.split)
        else:
            raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

        self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading  # TODO: maybe remove
        self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)


        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        model.names = {idx: label for idx, label in enumerate(model.labels)}  # TODO: mayby empty dict()
        self.init_metrics(model)
        self.jdict = []  # empty before each val
        for batch_i, batch in enumerate(bar):
            self.run_callbacks('on_val_batch_start')
            self.batch_i = batch_i
            im = cv2.imread(*batch["im_file"])
            res = model(im)
            # kek = YOLO('/home/wov/Downloads/model_api_data/ultralytics/yolov8n_openvino_model', "detect")
            # asdf = kek(im)


            import torch
            inputImgWidth = im.shape[1]
            inputImgHeight = im.shape[0]
            self.orig_width = 640
            self.orig_height = 640
            invertedScaleX, invertedScaleY = (
                inputImgWidth / self.orig_width,
                inputImgHeight / self.orig_height,
            )
            padLeft, padTop = 0, 0
            self.resize_type = "fit_to_window_letterbox"
            if (
                "fit_to_window" == self.resize_type
                or "fit_to_window_letterbox" == self.resize_type
            ):
                invertedScaleX = invertedScaleY = max(invertedScaleX, invertedScaleY)
                if "fit_to_window_letterbox" == self.resize_type:
                    padLeft = (self.orig_width - round(inputImgWidth / invertedScaleX)) // 2
                    padTop = (
                        self.orig_height - round(inputImgHeight / invertedScaleY)
                    ) // 2
            preds = [torch.tensor([(obj.xmin + padLeft, obj.ymin + padTop, obj.xmax + padLeft, obj.ymax + padTop, obj.score, obj.id) for obj in res.objects], dtype=torch.float32)]
            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)

            self.run_callbacks('on_val_batch_end')
        stats = self.get_stats()
        self.check_stats(stats)
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks('on_val_end')
        if self.training:
            model.float()
            results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix='val')}
            return {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
        else:
            LOGGER.info('Speed: %.1fms preprocess, %.1fms inference, %.1fms loss, %.1fms postprocess per image' %
                        tuple(self.speed.values()))
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / 'predictions.json'), 'w') as f:
                    LOGGER.info(f'Saving {f.name}...')
                    json.dump(self.jdict, f)  # flatten and save
                stats = self.eval_json(stats)  # update stats
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
            return stats


@pytest.mark.parametrize("impath", _impaths(all=True))
@pytest.mark.parametrize("pt", [Path("yolov8n.pt"), Path("yolov5n6u.pt"), ])
def test_config_detector(tmp_path, impath, pt):
    export_dir = _cached_config_models(pt)
    impl_wrapper = YOLOv5.create_model(export_dir / (pt.stem + ".xml"), device="CPU", configuration={"confidence_threshold": 0.001})
    kek = YOLO(export_dir, "detect")
    kek.overrides["imgsz"] = (impl_wrapper.w, impl_wrapper.h)

    validator = yolo.detect.DetectionValidator(save_dir=(Path(os.environ["DATA"]) / "ultralytics/tmp"), args={"data": "coco8.yaml", "imgsz": 640, "save_txt": True})
    mAP50_95 = validator(model=kek.model)["metrics/mAP50-95(B)"]
    print(mAP50_95)

    # model_api_validator = ModelAPIValidator(save_dir=(Path(os.environ["DATA"]) / "ultralytics/tmp"), args={"data": "coco8.yaml", "imgsz": 640, "save_txt": True})
    # mAP50_95 = model_api_validator(model=impl_wrapper)["metrics/mAP50-95(B)"]
    # print(mAP50_95)

    # mAP50_95 = kek.val(data="coco8.yaml").box.map
    # print(mAP50_95)

    # model = YOLO("/home/wov/Downloads/yolov8n.pt", "detect")
    # metrics = model.val(data="coco.yaml")
    # yolo = YOLOv5.create_model(export_dir / (pt.stem + ".xml"), configuration={"pad_value": 10, "resize_type": "asdf"}, device="CPU")
