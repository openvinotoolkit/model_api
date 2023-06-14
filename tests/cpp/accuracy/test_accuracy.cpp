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

#include <models/classification_model.h>
#include <models/detection_model.h>
#include <models/input_data.h>
#include <models/instance_segmentation.h>
#include <models/results.h>
#include <models/segmentation_model.h>
#include <adapters/openvino_adapter.h>

using json = nlohmann::json;

std::string PUBLIC_SCOPE_PATH = "../../tests/cpp/accuracy/public_scope.json";
std::string DATA_DIR = "../data";
std::string MODEL_PATH_TEMPLATE = "public/%s/FP16/%s.xml";

struct TestData {
    std::string image;
    std::vector<std::string> reference;
};

struct ModelData {
    std::string name;
    std::string type;
    std::vector<TestData> testData;
};

class ModelParameterizedTest : public testing::TestWithParam<ModelData> {
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

inline void from_json(const nlohmann::json& j, ModelData& test)
{
    test.name = j.at("name").get<std::string>();
    test.type = j.at("type").get<std::string>();
    for (auto& item : j.at("test_data")) {
        TestData data;
        data.image = item.at("image").get<std::string>();
        for (auto& ref : item.at("reference")) {
            data.reference.push_back(ref.get<std::string>());
        }
        test.testData.push_back(data);
    }
}

namespace {
std::vector<ModelData> GetTestData(const std::string& path)
{
    std::ifstream input(path);
    nlohmann::json j;
    input >> j;
    return j;
}
}

TEST_P(ModelParameterizedTest, AccuracyTest)
{
    auto modelData = GetParam();
    std::string modelPath;
    const std::string& name = modelData.name;
    if (name.substr(name.size() - 4) == ".xml") {
        modelPath = DATA_DIR + '/' + name;
    } else {
        modelPath = DATA_DIR + '/' + string_format(MODEL_PATH_TEMPLATE, name.c_str(), name.c_str());
    }
    const std::string& basename = modelPath.substr(modelPath.find_last_of("/\\") + 1);
    for (const std::string& modelXml: {modelPath, DATA_DIR + "/serialized/" + basename}) {
        if (modelData.type == "DetectionModel") {
            bool preload = true;
            auto model = DetectionModel::create_model(modelXml, {}, "", preload, "CPU");

            for (size_t i = 0; i < modelData.testData.size(); i++) {
                auto imagePath = DATA_DIR + "/" + modelData.testData[i].image;

                cv::Mat image = cv::imread(imagePath);
                if (!image.data) {
                    throw std::runtime_error{"Failed to read the image"};
                }

                auto result = model->infer(image);
                auto objects = result->objects;
                ASSERT_EQ(objects.size(), modelData.testData[i].reference.size());

                for (size_t j = 0; j < objects.size(); j++) {
                    std::stringstream prediction_buffer;
                    prediction_buffer << objects[j];
                    EXPECT_EQ(prediction_buffer.str(), modelData.testData[i].reference[j]);
                }
            }
        }
        else if (modelData.type == "ClassificationModel") {
            bool preload = true;
            auto model = ClassificationModel::create_model(modelXml, {}, preload, "CPU");

            for (size_t i = 0; i < modelData.testData.size(); i++) {
                ASSERT_EQ(modelData.testData[i].reference.size(), 1);
                auto imagePath = DATA_DIR + "/" + modelData.testData[i].image;

                cv::Mat image = cv::imread(imagePath);
                if (!image.data) {
                    throw std::runtime_error{"Failed to read the image"};
                }

                auto result = model->infer(image);
                std::cout << *result << ";;;" << modelData.testData[i].reference[0] << '\n';
                EXPECT_EQ(std::string{*result}, modelData.testData[i].reference[0]);
            }
        }
        else if (modelData.type == "SegmentationModel") {
            bool preload = true;
            auto model = SegmentationModel::create_model(modelXml, {}, preload, "CPU");

            for (size_t i = 0; i < modelData.testData.size(); i++) {
                auto imagePath = DATA_DIR + "/" + modelData.testData[i].image;

                cv::Mat image = cv::imread(imagePath);
                if (!image.data) {
                    throw std::runtime_error{"Failed to read the image"};
                }

                auto inference_result = model->infer(image)->asRef<ImageResultWithSoftPrediction>();

                cv::Mat predicted_mask[] = {inference_result.resultImage};
                int nimages = 1;
                int *channels = nullptr;
                cv::Mat mask;
                cv::Mat outHist;
                int dims = 1;
                int histSize[] = {256};
                float range[] = {0, 256};
                const float *ranges[] = {range};
                cv::calcHist(predicted_mask, nimages, channels, mask, outHist, dims, histSize, ranges);


                std::stringstream prediction_buffer;
                prediction_buffer << std::fixed << std::setprecision(3);
                for (int i = 0; i < range[1]; ++i) {
                    const float count = outHist.at<float>(i);
                    if (count > 0) {
                        prediction_buffer << i << ": " << count / predicted_mask[0].total() << ", ";
                    }
                }

                ASSERT_EQ(prediction_buffer.str(), modelData.testData[i].reference[0]);

                auto contours = model->getContours(inference_result);
                int j = 1; //First reference is histogram of mask
                for (auto &contour: contours) {
                    std::stringstream prediction_buffer;
                    prediction_buffer << contour;
                    ASSERT_EQ(prediction_buffer.str(), modelData.testData[i].reference[j]);
                    j++;
                }
            }
        }
        else if (modelData.type == "MaskRCNNModel") {
            bool preload = true;
            auto model = MaskRCNNModel::create_model(modelXml, {}, preload, "CPU");
            for (size_t i = 0; i < modelData.testData.size(); i++) {
                auto imagePath = DATA_DIR + "/" + modelData.testData[i].image;

                cv::Mat image = cv::imread(imagePath);
                if (!image.data) {
                    throw std::runtime_error{"Failed to read the image"};
                }
                const std::vector<SegmentedObject> objects = model->infer(image)->segmentedObjects;
                const std::vector<SegmentedObjectWithRects> withRects = add_rotated_rects(objects);
                ASSERT_EQ(withRects.size(), modelData.testData[i].reference.size());

                for (size_t j = 0; j < withRects.size(); j++) {
                    std::stringstream prediction_buffer;
                    prediction_buffer << withRects[j];
                    EXPECT_EQ(prediction_buffer.str(), modelData.testData[i].reference[j]);
                }

            }
        }
        else {
            throw std::runtime_error("Unknown model type: " + modelData.type);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(TestAccuracyPublic, ModelParameterizedTest, testing::ValuesIn(GetTestData(PUBLIC_SCOPE_PATH)));

class InputParser{
    public:
        InputParser (int &argc, char **argv){
            for (int i=1; i < argc; ++i)
                this->tokens.push_back(std::string(argv[i]));
        }

        const std::string& getCmdOption(const std::string &option) const{
            std::vector<std::string>::const_iterator itr;
            itr = std::find(this->tokens.begin(), this->tokens.end(), option);
            if (itr != this->tokens.end() && ++itr != this->tokens.end()){
                return *itr;
            }
            static const std::string empty_string("");
            return empty_string;
        }

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
