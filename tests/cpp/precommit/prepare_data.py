import argparse
import json
import os
from pathlib import Path
from urllib.request import urlopen, urlretrieve


def retrieve_otx_model(data_dir, model_name):
    destenation_folder = os.path.join(data_dir, "otx_models")
    os.makedirs(destenation_folder, exist_ok=True)
    urlretrieve(
        f"https://storage.openvinotoolkit.org/repositories/model_api/test/otx_models/{model_name}/openvino.xml",
        f"{destenation_folder}/{model_name}.xml",
    )
    urlretrieve(
        f"https://storage.openvinotoolkit.org/repositories/model_api/test/otx_models/{model_name}/openvino.bin",
        f"{destenation_folder}/{model_name}.bin",
    )


def prepare_model(
    data_dir="./data",
    public_scope=Path(__file__).resolve().parent / "public_scope.json",
):
    # TODO refactor this test so that it does not use eval
    # flake8: noqa: F401
    from model_api.models import ClassificationModel, DetectionModel, SegmentationModel

    with open(public_scope, "r") as f:
        public_scope = json.load(f)

    for model in public_scope:
        if model["name"].endswith(".xml"):
            continue
        model = eval(model["type"]).create_model(model["name"], download_dir=data_dir)


def prepare_data(data_dir="./data"):
    from io import BytesIO
    from zipfile import ZipFile

    COCO128_URL = "https://ultralytics.com/assets/coco128.zip"

    with urlopen(COCO128_URL) as zipresp:
        with ZipFile(BytesIO(zipresp.read())) as zfile:
            zfile.extractall(data_dir)

    urlretrieve(
        "https://raw.githubusercontent.com/Shenggan/BCCD_Dataset/master/BCCD/JPEGImages/BloodImage_00007.jpg",
        os.path.join(data_dir, "BloodImage_00007.jpg"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data and model preparate script")
    parser.add_argument(
        "-d",
        dest="data_dir",
        default="./data",
        help="Directory to store downloaded models and datasets",
    )
    parser.add_argument(
        "-p",
        dest="public_scope",
        default=Path(__file__).resolve().parent / "public_scope.json",
        help="JSON file with public model description",
    )

    args = parser.parse_args()

    prepare_model(args.data_dir, args.public_scope)
    prepare_data(args.data_dir)
    retrieve_otx_model(args.data_dir, "mlc_mobilenetv3_large_voc")
    retrieve_otx_model(args.data_dir, "tinynet_imagenet")
