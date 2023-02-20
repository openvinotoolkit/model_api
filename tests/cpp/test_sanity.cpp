#include <stddef.h>

#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/core.hpp>

#include <gtest/gtest.h>

#include <models/detection_model.h>
#include <models/input_data.h>
#include <models/results.h>

std::string MODEL_DIR="/home/alex/.cache/omz/public/ssd300/FP16/ssd300.xml";
std::string IMAGE_PATH="../tmp/coco128/images/train2017/000000000074.jpg";

TEST(test_sanity, simple_inference)
{
    //EXPECT_EQ(1000, 10 * std::stoi(argv[1]));	

    cv::Mat image = cv::imread(IMAGE_PATH);
    if (!image.data) {
        throw std::runtime_error{"Failed to read the image"};
    }

    auto model = DetectionModel::create_model(MODEL_DIR);
    auto result = model->infer(ImageInputData(image));
    ASSERT_TRUE(result->asRef<DetectionResult>().objects.size() > 0);
}