import tempfile

from openvino.model_api.models import Model


def test_detector_serialize():
    downloaded = Model.create_model(
        "ssd300", configuration={"mean_values": [0, 0, 0], "confidence_threshold": 0.6}
    )
    with tempfile.NamedTemporaryFile(suffix=".xml") as xml, tempfile.NamedTemporaryFile(
        suffix=".bin"
    ) as bin:
        downloaded.serialize(xml.name, bin.name)
        deserialized = Model.create_model(xml.name, weights_path=bin.name)
    assert type(downloaded) == type(deserialized)
    for attr in downloaded.parameters():
        assert getattr(downloaded, attr) == getattr(deserialized, attr)


def test_classifier_serialize():
    downloaded = Model.create_model(
        "efficientnet-b0-pytorch", configuration={"scale_values": [1, 1, 1], "topk": 6}
    )
    with tempfile.NamedTemporaryFile(suffix=".xml") as xml, tempfile.NamedTemporaryFile(
        suffix=".bin"
    ) as bin:
        downloaded.serialize(xml.name, bin.name)
        deserialized = Model.create_model(xml.name, weights_path=bin.name)
    assert type(downloaded) == type(deserialized)
    for attr in downloaded.parameters():
        assert getattr(downloaded, attr) == getattr(deserialized, attr)


def test_segmentor_serialize():
    downloaded = Model.create_model(
        "hrnet-v2-c1-segmentation",
        configuration={"reverse_input_channels": True, "labels": ["first", "second"]},
    )
    with tempfile.NamedTemporaryFile(suffix=".xml") as xml, tempfile.NamedTemporaryFile(
        suffix=".bin"
    ) as bin:
        downloaded.serialize(xml.name, bin.name)
        deserialized = Model.create_model(xml.name, weights_path=bin.name)
    assert type(downloaded) == type(deserialized)
    for attr in downloaded.parameters():
        assert getattr(downloaded, attr) == getattr(deserialized, attr)
