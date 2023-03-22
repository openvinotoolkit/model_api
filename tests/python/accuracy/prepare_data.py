import argparse
import os
from urllib.request import urlopen, urlretrieve


def retrieve_otx_model(data_dir, model_name):
    destenation_folder = os.path.join(data_dir, "otx_models")
    os.makedirs(destenation_folder, exist_ok=True)
    urlretrieve(
        f"https://storage.openvinotoolkit.org/repositories/model_api/test/otx_models/{model_name}/openvino.xml",
        f"{destenation_folder}/{model_name}.xml"
    )
    urlretrieve(
        f"https://storage.openvinotoolkit.org/repositories/model_api/test/otx_models/{model_name}/openvino.bin",
        f"{destenation_folder}/{model_name}.bin"
    )


def prepare_data(data_dir="./data"):
    from io import BytesIO
    from zipfile import ZipFile

    COCO128_URL = "https://ultralytics.com/assets/coco128.zip"

    with urlopen(COCO128_URL) as zipresp:
        with ZipFile(BytesIO(zipresp.read())) as zfile:
            zfile.extractall(data_dir)

    retrieve_otx_model(data_dir, "mlc_mobilenetv3_large_voc")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data and model preparate script")
    parser.add_argument(
        "-d",
        dest="data_dir",
        default="./data",
        help="Directory to store downloaded models and datasets",
    )

    args = parser.parse_args()

    prepare_data(args.data_dir)
