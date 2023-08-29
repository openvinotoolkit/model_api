import json
from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption("--data", action="store", help="data folder with dataset")
    parser.addoption(
        "--dump",
        action="store_true",
        default=False,
        help="whether to dump results into json file",
    )


def _impaths(data):
    impaths = sorted(
        file for file in (Path(data) / "coco128/images/train2017/").iterdir()
    )
    if not impaths:
        raise RuntimeError(f"{Path(data) / 'coco128/images/train2017/'} is empty")
    return impaths


def pytest_generate_tests(metafunc):
    if "pt" in metafunc.fixturenames:
        metafunc.parametrize(
            "pt",
            (
                "yolov5n6u.pt",
                "yolov5s6u.pt",
                "yolov5m6u.pt",
                "yolov5l6u.pt",
                "yolov5x6u.pt",
                "yolov8n.pt",
                "yolov8s.pt",
                "yolov8m.pt",
                "yolov8l.pt",
                "yolov8x.pt",
                "yolov5nu.pt",
                "yolov5su.pt",
                "yolov5mu.pt",
                "yolov5lu.pt",
                "yolov5xu.pt",
            ),
        )
    if "impath" in metafunc.fixturenames:
        metafunc.parametrize("impath", _impaths(metafunc.config.getoption("data")))


def pytest_configure(config):
    config.test_results = []


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    result = outcome.get_result()

    if result.when == "call":
        test_results = item.config.test_results
        with open("test_scope.json", "w") as outfile:
            json.dump(test_results, outfile, indent=4)
