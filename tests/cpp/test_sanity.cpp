#include <stddef.h>

#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <fstream>

#include <nlohmann/json.hpp>

#include <opencv2/core.hpp>

#include <gtest/gtest.h>

#include <models/detection_model.h>
#include <models/input_data.h>
#include <models/results.h>

namespace {

using json = nlohmann::json;

std::string PUBLIC_SCOPE_PATH = "../tests/cpp/public_scope.json";
std::string DATA_DIR = "../data";
std::string MODEL_PATH_TEMPLATE = "public/%s/FP16/%s.xml";
std::string IMAGE_PATH = "coco128/images/train2017/000000000074.jpg";

struct TestData {
    std::string name;
    std::string type;
};

class ModelParameterizedTest : public testing::TestWithParam<Test> {
};

template<typename... Args>
std::string string_format(const std::string &fmt, Args... args)
{
    size_t size = snprintf(nullptr, 0, fmt.c_str(), args...);
    std::string buf;
    buf.reserve(size + 1);
    buf.resize(size);
    snprintf(&buf[0], size + 1, fmt.c_str(), args...);
    return buf;
}

inline void from_json(const nlohmann::json& j, Test& test)
{
    test.name = j.at("name").get<std::string>();
    test.type = j.at("type").get<std::string>();
}
 
std::vector<TestData> GetTestData(const std::string& path)
{
    std::ifstream input(path);
    nlohmann::json j;
    input >> j;
    return j;
}
 
TEST_P(ModelParameterizedTest, SynchronousInference)
{
    cv::Mat image = cv::imread(DATA_DIR + "/" + IMAGE_PATH);
    if (!image.data) {
        throw std::runtime_error{"Failed to read the image"};
    }

    auto model_path = string_format(MODEL_PATH_TEMPLATE, GetParam().name.c_str(), GetParam().name.c_str());

    auto model = DetectionModel::create_model(DATA_DIR + "/" + model_path);
    auto result = model->infer(image);
    ASSERT_TRUE(result->objects.size() > 0);
}

} // namespace
 
INSTANTIATE_TEST_SUITE_P(TestSanityPublic, ModelParameterizedTest, testing::ValuesIn(GetTestData(PUBLIC_SCOPE_PATH)));

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);

    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <path_to_public_scope.json>" << std::endl;
        return 1;
    }

    DATA_DIR = argv[1];

    return RUN_ALL_TESTS();
}
