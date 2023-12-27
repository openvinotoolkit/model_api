from openvino.model_api.models import Model


def test_model_with_unnamed_output_load():
    # the model's output doesn't have a name
    _ = Model.create_model(
        "data/otx_models/tinynet_imagenet.xml",
        model_type="Classification",
        preload=True,
    )
