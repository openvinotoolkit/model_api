from pathlib import Path
import json
import argparse


def parepare_model(data_dir="./data", public_scope=Path(__file__).resolve().parent / "public_scope.json"):
    from openvino.model_api.models import Classification, DetectionModel, SegmentationModel
    
    with open(public_scope, "r") as f:
        public_scope = json.load(f)

    for model in public_scope:
        model = eval(model["type"]).create_model(model["name"], download_dir=data_dir)


def prepare_data(data_dir="./data"):
    from io import BytesIO
    from urllib.request import urlopen
    from zipfile import ZipFile

    COCO128_URL = "https://ultralytics.com/assets/coco128.zip"

    with urlopen(COCO128_URL) as zipresp:
        with ZipFile(BytesIO(zipresp.read())) as zfile:
            zfile.extractall(data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data and model preparate script")
    parser.add_argument("-d", dest="data_dir",
                        default="./data",
                        help="Directory to store downloaded models and datasets")
    parser.add_argument("-p", dest="public_scope",
                        default=Path(__file__).resolve().parent / "public_scope.json",
                        help="JSON file with public model description")

    args = parser.parse_args()
    
    parepare_model(args.data_dir, args.public_scope)
    prepare_data(args.data_dir)
