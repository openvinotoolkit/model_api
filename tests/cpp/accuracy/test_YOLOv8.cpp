#include <models/detection_model.h>
#include <models/input_data.h>
#include <models/results.h>

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

using namespace std;

namespace {
TEST(YOLOv8, Detector) {
    // Get data from env var, not form cmd arg to stay aligned with Python version
    const char* const data = getenv("DATA");
    ASSERT_NE(data, nullptr);
    const string& exported_path = string{data} + "/ultralytics/detectors/";
    for (const string model_name : {"yolov5mu_openvino_model", "yolov8l_openvino_model"}) {
        filesystem::path xml;
        for (auto const& dir_entry : filesystem::directory_iterator{exported_path + model_name}) {
            const filesystem::path& path = dir_entry.path();
            if (".xml" == path.extension()) {
                ASSERT_TRUE(xml.empty());
                xml = path;
            }
        }
        bool preload = true;
        unique_ptr<DetectionModel> yoloV8 = DetectionModel::create_model(xml.string(), {}, "", preload, "CPU");
        vector<filesystem::path> refpaths;
        for (auto const& dir_entry : filesystem::directory_iterator{exported_path + model_name + "/ref/"}) {
            refpaths.push_back(dir_entry.path());
        }
        ASSERT_GT(refpaths.size(), 0);
        sort(refpaths.begin(), refpaths.end());
        for (filesystem::path refpath : refpaths) {
            ifstream file{refpath};
            stringstream ss;
            ss << file.rdbuf();
            EXPECT_EQ(ss.str(), std::string{*yoloV8->infer(cv::imread(string{data} + "/coco128/images/train2017/" + refpath.stem().string() + ".jpg"))});
        }
    }
}
}
