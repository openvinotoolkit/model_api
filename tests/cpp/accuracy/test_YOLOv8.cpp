#include <models/detection_model.h>
#include <models/input_data.h>
#include <models/results.h>

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

namespace {
using namespace std;

string data() {
    // Get data from env var, not form cmd arg to stay aligned with Python version
    static const char* const data = getenv("DATA");
    EXPECT_NE(data, nullptr);
    return data;
}

struct Params {
    shared_ptr<DetectionModel> yolov8;
};

struct Model {
    shared_ptr<DetectionModel> yoloV8;
    string model_name;
    Model(const string& model_name) : model_name{model_name} {
        filesystem::path xml;
        for (auto const& dir_entry : filesystem::directory_iterator{data() + "/ultralytics/detectors/" + model_name}) {
            const filesystem::path& path = dir_entry.path();
            if (".xml" == path.extension()) {
                EXPECT_TRUE(xml.empty());
                xml = path;
            }
        }
        bool preload = true;
        yoloV8 = DetectionModel::create_model(xml.string(), {}, "", preload, "CPU");
    }
};

class AccurasySuit : public testing::TestWithParam<Model> {
};

TEST_P(AccurasySuit, TestDetector) {
    Model params = GetParam();
    vector<filesystem::path> refpaths;
    for (auto const& dir_entry : filesystem::directory_iterator{data() + "/ultralytics/detectors/" + params.model_name + "/ref/"}) {
        refpaths.push_back(dir_entry.path());
    }
    ASSERT_GT(refpaths.size(), 0);
    sort(refpaths.begin(), refpaths.end());
    for (filesystem::path refpath : refpaths) {
        ifstream file{refpath};
        stringstream ss;
        ss << file.rdbuf();
        EXPECT_EQ(ss.str(), string{*params.yoloV8->infer(cv::imread(data() + "/coco128/images/train2017/" + refpath.stem().string() + ".jpg"))});
    }
}

INSTANTIATE_TEST_SUITE_P(YOLOv8, AccurasySuit, testing::Values("yolov5mu_openvino_model", "yolov8l_openvino_model"));
}
