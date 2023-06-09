#include <models/detection_model.h>
#include <models/input_data.h>
#include <models/results.h>

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

namespace {
std::string DATA;

// TODO: test save-load
TEST(YOLOv5or8, Detector) {
    const std::string& exported_path = DATA + "YoloV8/exported/";
    std::filesystem::path xml;
    for (auto const& dir_entry : std::filesystem::directory_iterator{exported_path}) {
        const std::filesystem::path& path = dir_entry.path();
        if (".xml" == path.extension()) {
            if (!xml.empty()) {
                throw std::runtime_error(exported_path + " contain one .xml file");
            }
            xml = path;
        }
    }
    bool preload = true;
    std::unique_ptr<DetectionModel> yoloV8 = DetectionModel::create_model(xml, {}, "YoloV8", preload, "CPU");
    std::vector<std::filesystem::path> refpaths;  // TODO: prohibit empty ref folder
    for (auto const& dir_entry : std::filesystem::directory_iterator{DATA + "/YoloV8/exported/detector/ref/"}) {
        refpaths.push_back(dir_entry.path());
    }
    std::sort(refpaths.begin(), refpaths.end());
    for (std::filesystem::path refpath : refpaths) {
        const cv::Mat& im = cv::imread(DATA + "/coco128/images/train2017/" + refpath.stem().string() + ".jpg");
        std::vector<DetectedObject> objects = yoloV8->infer(im)->objects;
        std::ifstream file{refpath};
        std::string line;
        size_t i = 0;
        while (std::getline(file, line)) {
            ASSERT_LT(i, objects.size()) << refpath;
            std::stringstream prediction_buffer;
            prediction_buffer << objects[i];
            ASSERT_EQ(prediction_buffer.str(), line) << refpath;
            ++i;
        }
    }
}
}

int main(int argc, char *argv[]) {
    testing::InitGoogleTest(&argc, argv);
    if (2 != argc) {
        std::cerr << "Usage: " << argv[0] << " <path_to_data>\n";
        return 1;
    }
    DATA = argv[1];
    return RUN_ALL_TESTS();
}
