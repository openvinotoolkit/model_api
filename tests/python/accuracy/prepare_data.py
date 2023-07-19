import argparse
import os
from urllib.request import urlopen, urlretrieve


def retrieve_otx_model(data_dir, model_name):
    destenation_folder = os.path.join(data_dir, "otx_models")
    os.makedirs(destenation_folder, exist_ok=True)
    if "_onnx" in model_name:
        urlretrieve(
            f"https://storage.openvinotoolkit.org/repositories/model_api/test/otx_models/{model_name}/model.onnx",
            f"{destenation_folder}/{model_name}.onnx",
        )
    else:
        urlretrieve(
            f"https://storage.openvinotoolkit.org/repositories/model_api/test/otx_models/{model_name}/openvino.xml",
            f"{destenation_folder}/{model_name}.xml",
        )
        urlretrieve(
            f"https://storage.openvinotoolkit.org/repositories/model_api/test/otx_models/{model_name}/openvino.bin",
            f"{destenation_folder}/{model_name}.bin",
        )


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

    args = parser.parse_args()

    prepare_data(args.data_dir)
    retrieve_otx_model(args.data_dir, "mlc_mobilenetv3_large_voc")
    retrieve_otx_model(args.data_dir, "mlc_efficient_b0_voc")
    retrieve_otx_model(args.data_dir, "mlc_efficient_v2s_voc")
    retrieve_otx_model(args.data_dir, "det_mobilenetv2_atss_bccd")
    retrieve_otx_model(args.data_dir, "cls_mobilenetv3_large_cars")
    retrieve_otx_model(args.data_dir, "cls_mobilenetv3_large_cars_onnx")
    retrieve_otx_model(args.data_dir, "cls_efficient_b0_cars")
    retrieve_otx_model(args.data_dir, "cls_efficient_v2s_cars")
    retrieve_otx_model(args.data_dir, "Lite-hrnet-18")
    retrieve_otx_model(args.data_dir, "Lite-hrnet-18_mod2")
    retrieve_otx_model(args.data_dir, "Lite-hrnet-s_mod2")
    retrieve_otx_model(args.data_dir, "Lite-hrnet-x-mod3")
    retrieve_otx_model(args.data_dir, "is_efficientnetb2b_maskrcnn_coco_reduced")
    retrieve_otx_model(args.data_dir, "is_resnet50_maskrcnn_coco_reduced")
    retrieve_otx_model(args.data_dir, "mobilenet_v3_large_hc_cf")
    retrieve_otx_model(args.data_dir, "classification_model_with_xai_head")
    retrieve_otx_model(args.data_dir, "detection_model_with_xai_head")
    retrieve_otx_model(args.data_dir, "segmentation_model_with_xai_head")
    retrieve_otx_model(args.data_dir, "maskrcnn_model_with_xai_head")
    retrieve_otx_model(args.data_dir, "maskrcnn_xai_tiling")
    retrieve_otx_model(args.data_dir, "tile_classifier")
