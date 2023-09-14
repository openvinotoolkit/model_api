#include <models/detection_model.h>
#include <models/input_data.h>
#include <models/results.h>

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

using namespace std;

namespace {
string DATA;

TEST(YOLOv8, Detector) {
    const string& exported_path = DATA + "/ultralytics/detectors/";
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
        unique_ptr<DetectionModel> yoloV8 = DetectionModel::create_model(xml, {}, "", preload, "CPU");
        vector<filesystem::path> refpaths;  // TODO: prohibit empty ref folder
        for (auto const& dir_entry : filesystem::directory_iterator{DATA + "/ultralytics/detectors/" + model_name + "/ref/"}) {
            refpaths.push_back(dir_entry.path());
        }
        sort(refpaths.begin(), refpaths.end());
        for (filesystem::path refpath : refpaths) {
            const cv::Mat& im = cv::imread(DATA + "/coco128/images/train2017/" + refpath.stem().string() + ".jpg");
            ASSERT_NE(nullptr, im.data);
            ifstream file{refpath};
            stringstream ss;
            ss << file.rdbuf();
            EXPECT_EQ(ss.str(), std::string{*yoloV8->infer(im)});
        }
    }
}
}

int main(int argc, char *argv[]) {
    testing::InitGoogleTest(&argc, argv);
    if (2 != argc) {
        cerr << "Usage: " << argv[0] << " <path_to_data>\n";
        return 1;
    }
    DATA = argv[1];
    return RUN_ALL_TESTS();
}
