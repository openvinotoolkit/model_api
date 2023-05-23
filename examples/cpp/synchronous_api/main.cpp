/*
// Copyright (C) 2018-2022 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include <stddef.h>

#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>

#include <models/segmentation_model.h>
#include <models/input_data.h>
#include <models/results.h>


int main(int argc, char* argv[]) {
    try {
        if (argc != 3) {
            std::cerr << "Usage : " << argv[0] << " <path_to_model> <path_to_image>"
                      << std::endl;
            return EXIT_FAILURE;
        }

        cv::Mat image = cv::imread(argv[2]);
        if (!image.data) {
            throw std::runtime_error{"Failed to read the image"};
        }

        // Instantiate Object Detection model
        auto model = SegmentationModel::create_model(argv[1]);
        // Run the inference
        auto result = std::shared_ptr<ImageResultWithSoftPrediction>(static_cast<ImageResultWithSoftPrediction*>(model->infer(image).release()));

        auto contours = model->getContours(result);

        auto output_image = image.clone();

        std::vector<std::vector<cv::Point>> cv_contours = {};
        for (auto &contour: contours) {
            cv_contours.push_back(contour.shape);
        }

        cv::drawContours(output_image, cv_contours, -1, 255, 1);
        cv::imwrite("/data/output.png", output_image);
        std::cout << contours.size() << std::endl;


    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown/internal exception happened." << std::endl;
        return 1;
    }

    return 0;
}
