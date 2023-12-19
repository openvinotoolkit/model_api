from openvino.model_api.models import Model


def test_detector_save():
    # the model's output doesn't have a name
    _ = Model.create_model(
        "data/otx_models/tinynet_test.xml", model_type="Classification", preload=True
    )
