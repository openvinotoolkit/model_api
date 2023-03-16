import tempfile

from openvino.model_api.models import Model

configurationImageModel = (
    "mean_values",
    "scale_values",
    "reverse_input_channels",
    "resize_type",
    "embed_preprocessing",
)


def test_detector_serialize():
    downloaded = Model.create_model(
        "ssd300", configuration={"mean_values": [0, 0, 0], "confidence_threshold": 0.6}
    )
    with tempfile.NamedTemporaryFile(suffix=".xml") as xml, tempfile.NamedTemporaryFile(
        suffix=".bin"
    ) as bin:
        downloaded.serialize(xml.name, bin.name)
        restored = Model.create_model(xml.name, weights_path=bin.name)
    assert type(downloaded) == type(restored)
    for attr in configurationImageModel + (
        "confidence_threshold",
        "labels",
        "path_to_labels",
    ):
        assert getattr(downloaded, attr) == getattr(restored, attr)


def test_classifier_serialize():
    downloaded = Model.create_model(
        "efficientnet-b0-pytorch", configuration={"scale_values": [1, 1, 1], "topk": 6}
    )
    with tempfile.NamedTemporaryFile(suffix=".xml") as xml, tempfile.NamedTemporaryFile(
        suffix=".bin"
    ) as bin:
        downloaded.serialize(xml.name, bin.name)
        restored = Model.create_model(xml.name, weights_path=bin.name)
    assert type(downloaded) == type(restored)
    for attr in configurationImageModel + ("topk", "labels", "path_to_labels"):
        assert getattr(downloaded, attr) == getattr(restored, attr)


def test_segmentor_serialize():
    downloaded = Model.create_model(
        "hrnet-v2-c1-segmentation",
        configuration={"reverse_input_channels": True, "labels": ["first", "second"]},
    )
    with tempfile.NamedTemporaryFile(suffix=".xml") as xml, tempfile.NamedTemporaryFile(
        suffix=".bin"
    ) as bin:
        downloaded.serialize(xml.name, bin.name)
        restored = Model.create_model(xml.name, weights_path=bin.name)
    assert type(downloaded) == type(restored)
    for attr in configurationImageModel + ("labels", "path_to_labels"):
        assert getattr(downloaded, attr) == getattr(restored, attr)
