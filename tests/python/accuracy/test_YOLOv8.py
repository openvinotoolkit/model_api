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
import ultralytics
import torch
import types




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


from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.utils import LOGGER, TQDM, callbacks, colorstr, emojis

torch.inference_mode

class ModelAPIValidator(yolo.detect.DetectionValidator):
    def __init__(self, args):
        super().__init__(args=args)

    def __call__(self, model, impl):
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
        # self.dataloader.dataset.transforms.transforms = self.dataloader.dataset.transforms.transforms[1:]
        # self.dataloader.dataset.transforms.transforms = (lambda di: {
        #     'batch_idx': torch.zeros(len(di['instances'])),
        #     'bboxes': torch.from_numpy(di['instances'].bboxes),
        #     'cls': torch.from_numpy(di['cls']),
        #     'img': torch.from_numpy(di['img'].transpose(2, 0, 1)),
        #     'im_file': di['im_file'],
        #     'ori_shape': (640, 640),
        #     'ratio_pad': [(1.0,), (0.0, 0.0)],
        # },)
        class asdf:
            def __init__(self):
                self.names = {idx: label for idx, label in enumerate(impl.labels)}
        # model.names = {idx: label for idx, label in enumerate(impl.labels)}  # TODO: meayby empty dict()
        self.init_metrics(asdf())
        self.jdict = []  # empty before each val
        for batch_i, batch in enumerate(self.dataloader):
            self.run_callbacks('on_val_batch_start')
            self.batch_i = batch_i
            # im = batch['img'][0].numpy().transpose(1, 2, 0)
            im = cv2.imread(*batch['im_file'])
            # batch['img'] = torch.from_numpy(im.transpose(2, 0, 1)[None])
            pred = model.predict(im, conf=0.001)

            # self.orig_width = 640
            # self.orig_height = 640

            # scale = min(self.orig_width / im.shape[1], self.orig_height / im.shape[0])
            # preds = torch.tensor([(obj.xmin, obj.ymin, obj.xmax, obj.ymax, obj.score, obj.id) for obj in pred.objects], dtype=torch.float32)
            preds = pred[0].boxes.data
            # batch['bboxes'] = batch['bboxes'] * torch.tensor((640, 640, 640, 640))
            # batch['ori_shape'] = [(640, 640),]
            # img1_shape = 640, 640
            # img0_shape = im.shape[1], im.shape[0]
            # gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            # pad = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1), round(
            #     (img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1)  # wh padding
            preds[:, :4] *= batch['ratio_pad'][0][0][0]
            preds[:, :4] += torch.tensor([batch['ratio_pad'][0][1][0], batch['ratio_pad'][0][1][1], batch['ratio_pad'][0][1][0], batch['ratio_pad'][0][1][1]], dtype=torch.int)
            # batch['ratio_pad'] = ([batch['ratio_pad'][0], pad],)
            # batch['ori_shape'] = ([im.shape[1], im.shape[0]],)
            self.update_metrics([preds], batch)
            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, [preds], batch_i)

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
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / 'predictions.json'), 'w') as f:
                    LOGGER.info(f'Saving {f.name}...')
                    # json.dump(self.jdict, f)  # flatten and save
                stats = self.eval_json(stats)  # update stats
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
            return stats


class Format:
    def __call__(self, labels):
        """Return formatted image, classes, bounding boxes & keypoints to be used by 'collate_fn'."""
        img = labels.pop('img')
        h, w = img.shape[:2]
        instances = labels['instances']
        instances.convert_bbox(format='xywh')
        instances.normalize(w, h)
        labels['img'] = img
        return labels


import cv2
class ModelAPIValidator2(yolo.detect.DetectionValidator):
    def __init__(self, args):
        super().__init__(args=args)

    def __call__(self, yolo, names):
        self.data = check_det_dataset(self.args.data)
        dataloader = self.get_dataloader(self.data[self.args.split], batch_size=1)
        dataloader.dataset.transforms.transforms = [dataloader.dataset.transforms.transforms[0], Format(), lambda di: {
            'batch_idx': torch.zeros(len(di['instances'])),
            'bboxes': torch.from_numpy(di['instances'].bboxes),
            'cls': torch.from_numpy(di['cls']),
            'img': torch.from_numpy(di['img'].transpose(2, 0, 1)),  # [::-1]
            'im_file': di['im_file'],
            'ori_shape': di['ori_shape'],
            'ratio_pad': di['ratio_pad'],
        }]
        self.init_metrics(names)
        for batch in dataloader:
            my_pred = yolo.predict(cv2.imread(*batch['im_file']), conf=0.001)
            preds = my_pred[0].boxes.data.clone()
            scale_w, scale_h = batch['ratio_pad'][0][0]
            pad_w = (640 - round(batch['ori_shape'][0][1] * scale_w)) // 2
            pad_h = (640 - round(batch['ori_shape'][0][0] * scale_h)) // 2
            preds[:, (0, 2)] *= scale_w
            preds[:, (1, 3)] *= scale_h
            preds[:, (0, 2)] += pad_w
            preds[:, (1, 3)] += pad_h
            try:
                assert (pad_w, pad_h) == batch['ratio_pad'][0][1]
            except:
                breakpoint()
            batch['ratio_pad'] = [([scale_w, scale_h], [pad_w, pad_h])]
            self.update_metrics([preds], batch)
        return self.get_stats()


@pytest.mark.parametrize("pt", [Path("yolov8n.pt")])#, Path("yolov5n6u.pt")])
def test_detector_metric(pt):
    export_dir = export_dir = Path(
        YOLO(Path(os.environ["DATA"]) / "ultralytics" / pt, "detect").export(format="openvino", half=True)
    )
    impl_wrapper = YOLOv5.create_model(export_dir / pt.with_suffix(".xml"), device="CPU", configuration={"confidence_threshold": 0.001})

    kek = YOLO(export_dir, "detect")
    kek.overrides["imgsz"] = (impl_wrapper.w, impl_wrapper.h)  # TODO: wh or hw?

    mAP50_95 = ModelAPIValidator2(args={"data": "coco128.yaml", "imgsz": impl_wrapper.w})(kek, types.SimpleNamespace(names={idx: label for idx, label in enumerate(impl_wrapper.labels)}))["metrics/mAP50-95(B)"]
    print(f"{mAP50_95:.99f}")
    assert mAP50_95 >= 0.453236284183963278326956469754804857075214385986328125000000000000000000000000000000000000000000000
    # 0.453149872092887873176181301460019312798976898193359375000000000000000000000000000000000000000000000
    # AUTO: 0.450826465645132734572086974367266520857810974121093750000000000000000000000000000000000000000000000
    mAP50_95 = ModelAPIValidator2(args={"data": "coco.yaml", "imgsz": impl_wrapper.w})(kek, types.SimpleNamespace(names={idx: label for idx, label in enumerate(impl_wrapper.labels)}))["metrics/mAP50-95(B)"]
    print(f"{mAP50_95:.99f}")
    assert mAP50_95 >= 0.370558668805542223978477522905450314283370971679687500000000000000000000000000000000000000000000000
    # 0.370352469463463451759821509767789393663406372070312500000000000000000000000000000000000000000000000
    # AUTO: 0.369873965487229283688463965518167242407798767089843750000000000000000000000000000000000000000000000
    # 0.370148340185503577082215542759513482451438903808593750000000000000000000000000000000000000000000000

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
