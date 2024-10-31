#
# Copyright (C) 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from model_api.models import Model


def test_detector_save(tmp_path):
    downloaded = Model.create_model(
        "ssd_mobilenet_v1_fpn_coco",
        configuration={"mean_values": [0, 0, 0], "confidence_threshold": 0.6},
    )
    assert (
        downloaded.get_model()
        .get_rt_info(["model_info", "embedded_processing"])
        .astype(bool)
    )
    xml_path = str(tmp_path / "a.xml")
    downloaded.save(xml_path)
    deserialized = Model.create_model(xml_path)
    assert type(downloaded) is type(deserialized)
    for attr in downloaded.parameters():
        assert getattr(downloaded, attr) == getattr(deserialized, attr)


def test_classifier_save(tmp_path):
    downloaded = Model.create_model(
        "efficientnet-b0-pytorch", configuration={"scale_values": [1, 1, 1], "topk": 6}
    )
    assert (
        downloaded.get_model()
        .get_rt_info(["model_info", "embedded_processing"])
        .astype(bool)
    )
    xml_path = str(tmp_path / "a.xml")
    downloaded.save(xml_path)
    deserialized = Model.create_model(xml_path)
    assert type(downloaded) is type(deserialized)
    for attr in downloaded.parameters():
        assert getattr(downloaded, attr) == getattr(deserialized, attr)


def test_segmentor_save(tmp_path):
    downloaded = Model.create_model(
        "hrnet-v2-c1-segmentation",
        configuration={"reverse_input_channels": True, "labels": ["first", "second"]},
    )
    assert (
        downloaded.get_model()
        .get_rt_info(["model_info", "embedded_processing"])
        .astype(bool)
    )
    xml_path = str(tmp_path / "a.xml")
    downloaded.save(xml_path)
    deserialized = Model.create_model(xml_path)
    assert type(downloaded) is type(deserialized)
    for attr in downloaded.parameters():
        assert getattr(downloaded, attr) == getattr(deserialized, attr)
