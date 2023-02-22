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

using json = nlohmann::json;

std::string MODEL_PATH_TEMPLATE = "../tmp/public/%s/FP16/%s.xml";
std::string IMAGE_PATH = "../tmp/coco128/images/train2017/000000000074.jpg";

struct Test {
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
 
std::vector<Test> GetTests(const std::string& path)
{
    std::ifstream input(path);
    nlohmann::json j;
    input >> j;
    return j;
}
 
TEST_P(ModelParameterizedTest, SynchronousInference)
{
    cv::Mat image = cv::imread(IMAGE_PATH);
    if (!image.data) {
        throw std::runtime_error{"Failed to read the image"};
    }

    auto model_path = string_format(MODEL_PATH_TEMPLATE, GetParam().name.c_str(), GetParam().name.c_str());

    auto model = DetectionModel::create_model(model_path);
    auto result = model->infer(ImageInputData(image));
    ASSERT_TRUE(result->asRef<DetectionResult>().objects.size() > 0);
}
 
INSTANTIATE_TEST_SUITE_P(TestSanity, ModelParameterizedTest, testing::ValuesIn(GetTests("../input.json")));
