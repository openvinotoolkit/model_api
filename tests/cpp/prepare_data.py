CACHE_DIR = "./tmp/"


def parepare_model():
    from openvino.model_api.models import DetectionModel

    ssd = DetectionModel.create_model("ssd300", cache_dir=CACHE_DIR)


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
