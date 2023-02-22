

CACHE_DIR = "./tmp/"
PUBLIC_SCOPE_PATH = "./public_scope.json"


def parepare_model():
    import json
    from openvino.model_api.models import DetectionModel
    
    with open(PUBLIC_SCOPE_PATH, "r") as f:
        public_scope = json.load(f)
    
    for model in public_scope:
        model = eval(model["type"]).create_model(model["name"], download_dir=CACHE_DIR)


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
