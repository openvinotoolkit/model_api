from openvino.model_api.models import Model


def test_detector_save(tmp_path):
    downloaded = Model.create_model(
        "ssd300", configuration={"mean_values": [0, 0, 0], "confidence_threshold": 0.6}
    )
    assert True == downloaded.get_model().get_rt_info(["model_info", "embedded_processing"]).astype(bool)
    xml_path = str(tmp_path / "a.xml")
    downloaded.save(xml_path)
    deserialized = Model.create_model(xml_path)
    assert type(downloaded) == type(deserialized)
    for attr in downloaded.parameters():
        assert getattr(downloaded, attr) == getattr(deserialized, attr)


def test_classifier_save(tmp_path):
    downloaded = Model.create_model(
        "efficientnet-b0-pytorch", configuration={"scale_values": [1, 1, 1], "topk": 6}
    )
    assert True == downloaded.get_model().get_rt_info(["model_info", "embedded_processing"]).astype(bool)
    xml_path = str(tmp_path / "a.xml")
    downloaded.save(xml_path)
    deserialized = Model.create_model(xml_path)
    assert type(downloaded) == type(deserialized)
    for attr in downloaded.parameters():
        assert getattr(downloaded, attr) == getattr(deserialized, attr)


def test_segmentor_save(tmp_path):
    downloaded = Model.create_model(
        "hrnet-v2-c1-segmentation",
        configuration={"reverse_input_channels": True, "labels": ["first", "second"]},
    )
    assert True == downloaded.get_model().get_rt_info(["model_info", "embedded_processing"]).astype(bool)
    xml_path = str(tmp_path / "a.xml")
    downloaded.save(xml_path)
    deserialized = Model.create_model(xml_path)
    assert type(downloaded) == type(deserialized)
    for attr in downloaded.parameters():
        assert getattr(downloaded, attr) == getattr(deserialized, attr)
