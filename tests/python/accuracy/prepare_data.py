import argparse
import json
from pathlib import Path


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

    prepare_data(args.data_dir)
