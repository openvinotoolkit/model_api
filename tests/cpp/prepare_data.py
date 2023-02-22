from openvino.model_api.models import DetectionModel

CACHE_DIR = "./tmp/"

PUBLIC_MODELS = {
    "ssd300": DetectionModel,
    "ssd_mobilenet_v1_fpn_coco": DetectionModel,
    "ssdlite_mobilenet_v2": DetectionModel,
}


def parepare_model():
    for name, class_name in PUBLIC_MODELS.items():
        model = class_name.create_model(name, download_dir=CACHE_DIR)


def prepare_data():
    from io import BytesIO
    from urllib.request import urlopen
    from zipfile import ZipFile

    COCO128_URL = "https://ultralytics.com/assets/coco128.zip"

    with urlopen(COCO128_URL) as zipresp:
        with ZipFile(BytesIO(zipresp.read())) as zfile:
            zfile.extractall(CACHE_DIR)


if __name__ == "__main__":
    parepare_model()
    prepare_data()
