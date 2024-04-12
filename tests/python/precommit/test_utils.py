import numpy as np

from model_api.adapters import OpenvinoAdapter, create_core
from model_api.adapters.utils import resize_image_with_aspect_ocv


def test_resize_image_with_aspect_ocv():
    model_adapter = OpenvinoAdapter(
        core=create_core(),
        model="data/otx_models/tinynet_imagenet.xml",  # refer test_load.py
        weights_path="data/otx_models/tinynet_imagenet.bin",  # refer test_load.py
        device="CPU",
        max_num_requests=1,
        plugin_config={"PERFORMANCE_HINT": "THROUGHPUT"},
    )

    model_adapter.embed_preprocessing(
        layout="NCHW",
        resize_mode="fit_to_window",
        interpolation_mode="LINEAR",
        target_shape=(1024, 1024),
        pad_value=0,
        brg2rgb=False,
    )

    img = np.ones((256, 512, 3), dtype=np.uint8)
    ov_results = model_adapter.compiled_model(img[None])
    ov_results = list(ov_results.values())[0][0].transpose(1, 2, 0).astype(np.uint8)
    np_results = resize_image_with_aspect_ocv(img, (1024, 1024))

    assert np.sum(np.abs(ov_results - np_results)) < 1e-05
