#include <stddef.h>

#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <fstream>
#include <cstdio>

#include <nlohmann/json.hpp>

#include <opencv2/core.hpp>

#include <gtest/gtest.h>

#include <models/classification_model.h>
#include <models/detection_model.h>
#include <models/detection_model_ssd.h>
#include <models/input_data.h>
#include <models/results.h>
#include <adapters/openvino_adapter.h>

using json = nlohmann::json;

namespace{
std::string DATA_DIR;
std::string MODEL_PATH_TEMPLATE = "public/%s/FP16/%s.xml";
std::string IMAGE_PATH = "coco128/images/train2017/000000000074.jpg";

std::string TMP_MODEL_FILE = "tmp_model.xml";

struct ModelData {
    std::string name;
    ModelData(const std::string& name)
        : name(name) {}
};

class MockAdapter : public OpenVINOInferenceAdapter {
    public:
        MockAdapter(const std::string& modelPath)
            : OpenVINOInferenceAdapter() {
            ov::Core core;
            auto model = core.read_model(modelPath);
            loadModel(model, core, "CPU");
            std::cout << "QQQQQQQQQQQQQQQQQ\n";
        }
};

class ClassificationModelParameterizedTest : public testing::TestWithParam<ModelData> {
};

class SSDModelParameterizedTest : public testing::TestWithParam<ModelData> {
};

class ClassificationModelParameterizedTestSaveLoad : public testing::TestWithParam<ModelData> {
    protected:
        void TearDown() override {
            auto fileName = TMP_MODEL_FILE;
            std::remove(fileName.c_str());
            std::remove(fileName.replace(fileName.end() - 4, fileName.end(), ".bin").c_str());
        }
};

class DetectionModelParameterizedTestSaveLoad : public ClassificationModelParameterizedTestSaveLoad {
};

template<typename... Args>
std::string string_format(const std::string &fmt, Args... args) {
    size_t size = snprintf(nullptr, 0, fmt.c_str(), args...);
    std::string buf;
    buf.reserve(size + 1);
    buf.resize(size);
    snprintf(&buf[0], size + 1, fmt.c_str(), args...);
    return buf;
}

TEST_P(DetectionModelParameterizedTestSaveLoad, TestDetctionCorrectnessAfterSaveLoadWithAdapter) {
    cv::Mat image = cv::imread(DATA_DIR + "/" + IMAGE_PATH);
    if (!image.data) {
        throw std::runtime_error{"Failed to read the image"};
    }

    auto model_path = string_format(MODEL_PATH_TEMPLATE, GetParam().name.c_str(), GetParam().name.c_str());
    std::cout << "CCCCCCCCCCCCCCC\n";
    auto model = DetectionModel::create_model(DATA_DIR + "/" + model_path);
    std::cout << "DDDDDDDDDDDDDDDDDDDDD\n";
    auto ov_model = model->getModel();
    std::cout << "EEEEEEEEEEEEEEEEEee\n";
    ov::serialize(ov_model, TMP_MODEL_FILE);
    std::cout << "FFFFFFFFFFFFFFF\n";
    auto result = model->infer(image)->objects;

    std::cout << "AAAAA\n";
    ov::Core core;
    auto ovmodel = core.read_model(TMP_MODEL_FILE);
    std::cout << "BBBBBBB\n";
    auto compiledModel = core.compile_model(ovmodel, "CPU", {});
    std::cout << "GGGGGGGGG\n";
    std::shared_ptr<InferenceAdapter> adapter = std::make_shared<MockAdapter>(TMP_MODEL_FILE);
    std::cout << "HHHHHHHHHHHHHH\n";
}

INSTANTIATE_TEST_SUITE_P(SSDTestInstance, DetectionModelParameterizedTestSaveLoad, ::testing::Values(ModelData("ssd_mobilenet_v1_fpn_coco")));

char* parse_arg(int argc, char * const argv[], const char arg[]) {
    for (int i = 1; i < argc - 1; ++i) {
        if (!strcmp(argv[i], arg)) {
            return argv[i + 1];
        }
    }
    std::cerr << "Missing " << arg << ". Usage: " << argv[0] << " -d <path_to_data_dir>\n";
    exit(1);
}
}

int main(int argc, char *argv[]) {
    DATA_DIR = parse_arg(argc, argv, "-d");
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
