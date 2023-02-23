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

std::string PUBLIC_SCOPE_PATH = "../tests/cpp/public_scope.json";
std::string DATA_DIR = "../data";
std::string MODEL_PATH_TEMPLATE = "public/%s/FP16/%s.xml";
std::string IMAGE_PATH = "coco128/images/train2017/000000000074.jpg";

struct TestData {
    std::string name;
    std::string type;
};

class ModelParameterizedTest : public testing::TestWithParam<TestData> {
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

inline void from_json(const nlohmann::json& j, TestData& test)
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
 
INSTANTIATE_TEST_SUITE_P(TestSanityPublic, ModelParameterizedTest, testing::ValuesIn(GetTestData(PUBLIC_SCOPE_PATH)));

class InputParser{
    public:
        InputParser (int &argc, char **argv){
            for (int i=1; i < argc; ++i)
                this->tokens.push_back(std::string(argv[i]));
        }
        /// @author iain
        const std::string& getCmdOption(const std::string &option) const{
            std::vector<std::string>::const_iterator itr;
            itr =  std::find(this->tokens.begin(), this->tokens.end(), option);
            if (itr != this->tokens.end() && ++itr != this->tokens.end()){
                return *itr;
            }
            static const std::string empty_string("");
            return empty_string;
        }
        /// @author iain
        bool cmdOptionExists(const std::string &option) const{
            return std::find(this->tokens.begin(), this->tokens.end(), option)
                   != this->tokens.end();
        }
    private:
        std::vector <std::string> tokens;
};

void print_help(const char* program_name) 
{
    std::cout << "Usage: " << program_name << " -p <path_to_public_scope.json> -d <path_to_data>" << std::endl;
}

int main(int argc, char **argv) 
{
    InputParser input(argc, argv);

    if(input.cmdOptionExists("-h")){
        print_help(argv[0]);
        return 1;
    }
    const std::string &public_scope = input.getCmdOption("-p");
    if (!public_scope.empty()){
        PUBLIC_SCOPE_PATH = public_scope;
    }
    else{
        print_help(argv[0]);
        return 1;
    }
    const std::string &data_dir = input.getCmdOption("-d");
    if (!data_dir.empty()){
        DATA_DIR = data_dir;
    }
    else{
        print_help(argv[0]);
        return 1;
    }

    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
