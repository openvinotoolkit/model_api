from openvino.model_api.models import Model


def test_detector_save(tmp_path):
    downloaded = Model.create_model(
        "ssd300", configuration={"mean_values": [0, 0, 0], "confidence_threshold": 0.6}
    )
    embedded_processing = downloaded.get_model().get_rt_info(
        ["model_info", "embedded_processing"]
    )
    if type(embedded_processing) != bool:
        # TODO: uncomment after update to 2023.0
        # 2023.0 return OVAny which needs to be casted with astype()
        embedded_processing = embedded_processing.astype(bool)
    assert True == embedded_processing
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
    if type(embedded_processing) != bool:
        # TODO: uncomment after update to 2023.0
        # 2023.0 return OVAny which needs to be casted with astype()
        embedded_processing = embedded_processing.astype(bool)
    assert True == embedded_processing
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
    if type(embedded_processing) != bool:
        # TODO: uncomment after update to 2023.0
        # 2023.0 return OVAny which needs to be casted with astype()
        embedded_processing = embedded_processing.astype(bool)
    assert True == embedded_processing
    xml_path = str(tmp_path / "a.xml")
    downloaded.save(xml_path)
    deserialized = Model.create_model(xml_path)
    assert type(downloaded) == type(deserialized)
    for attr in downloaded.parameters():
        assert getattr(downloaded, attr) == getattr(deserialized, attr)
