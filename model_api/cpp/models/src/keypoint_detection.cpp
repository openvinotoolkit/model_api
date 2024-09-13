/*
// Copyright (C) 2024 Intel Corporation
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

#include "models/keypoint_detection.h"

#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>

#include "models/input_data.h"
#include "models/internal_model_data.h"
#include "models/results.h"
#include "utils/slog.hpp"

namespace {

}

std::string KeypointDetectionModel::ModelType = "keypoint_detection";

void KeypointDetectionModel::init_from_config(const ov::AnyMap& top_priority, const ov::AnyMap& mid_priority) {
    labels = get_from_any_maps("labels", top_priority, mid_priority, labels);
}

KeypointDetectionModel::KeypointDetectionModel(std::shared_ptr<ov::Model>& model, const ov::AnyMap& configuration) : ImageModel(model, configuration) {
    init_from_config(configuration, model->has_rt_info("model_info") ? model->get_rt_info<ov::AnyMap>("model_info") : ov::AnyMap{});
}

KeypointDetectionModel::KeypointDetectionModel(std::shared_ptr<InferenceAdapter>& adapter, const ov::AnyMap& configuration)
        : ImageModel(adapter, configuration) {
    init_from_config(configuration, adapter->getModelConfig());
}

std::unique_ptr<KeypointDetectionModel> KeypointDetectionModel::create_model(const std::string& modelFile, const ov::AnyMap& configuration, bool preload, const std::string& device) {
    auto core = ov::Core();
    std::shared_ptr<ov::Model> model = core.read_model(modelFile);

    // Check model_type in the rt_info, ignore configuration
    std::string model_type = KeypointDetectionModel::ModelType;
    try {
        if (model->has_rt_info("model_info", "model_type") ) {
            model_type = model->get_rt_info<std::string>("model_info", "model_type");
        }
    } catch (const std::exception&) {
        slog::warn << "Model type is not specified in the rt_info, use default model type: " << model_type << slog::endl;
    }

    if (model_type != KeypointDetectionModel::ModelType) {
        throw std::runtime_error("Incorrect or unsupported model_type is provided in the model_info section: " + model_type);
    }

    std::unique_ptr<KeypointDetectionModel> kp_detector{new KeypointDetectionModel(model, configuration)};
    kp_detector->prepare();
    if (preload) {
        kp_detector->load(core, device);
    }
    return kp_detector;
}

std::unique_ptr<KeypointDetectionModel> KeypointDetectionModel::create_model(std::shared_ptr<InferenceAdapter>& adapter) {
    const ov::AnyMap& configuration = adapter->getModelConfig();
    auto model_type_iter = configuration.find("model_type");
    std::string model_type = KeypointDetectionModel::ModelType;
    if (model_type_iter != configuration.end()) {
        model_type = model_type_iter->second.as<std::string>();
    }

    if (model_type != KeypointDetectionModel::ModelType) {
        throw std::runtime_error("Incorrect or unsupported model_type is provided: " + model_type);
    }

    std::unique_ptr<KeypointDetectionModel> kp_detector{new KeypointDetectionModel(adapter)};
    return kp_detector;
}

void KeypointDetectionModel::updateModelInfo() {
    ImageModel::updateModelInfo();

    model->set_rt_info(KeypointDetectionModel::ModelType, "model_info", "model_type");
    model->set_rt_info(labels, "model_info", "labels");
}

void KeypointDetectionModel::prepareInputsOutputs(std::shared_ptr<ov::Model>& model) {
    // --------------------------- Configure input & output ---------------------------------------------
    // --------------------------- Prepare input  -----------------------------------------------------
    if (model->inputs().size() != 1) {
        throw std::logic_error(KeypointDetectionModel::ModelType + " model wrapper supports topologies with only 1 input");
    }
    const auto& input = model->input();
    inputNames.push_back(input.get_any_name());
    const ov::Layout& inputLayout = getInputLayout(input);
    const ov::Shape& inputShape = input.get_partial_shape().get_max_shape();
    if (inputShape.size() != 4 || inputShape[ov::layout::channels_idx(inputLayout)] != 3) {
        throw std::logic_error("3-channel 4-dimensional model's input is expected");
    }

    if (model->outputs().size() != 2) {
        throw std::logic_error(KeypointDetectionModel::ModelType + " model wrapper supports topologies with 2 outputs");
    }

    if (!embedded_processing) {
        model = ImageModel::embedProcessing(model,
                                        inputNames[0],
                                        inputLayout,
                                        resizeMode,
                                        interpolationMode,
                                        ov::Shape{inputShape[ov::layout::width_idx(inputLayout)],
                                                  inputShape[ov::layout::height_idx(inputLayout)]},
                                        pad_value,
                                        reverse_input_channels,
                                        {},
                                        scale_values);

        ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);
        model = ppp.build();
        embedded_processing = true;
        useAutoResize = true;
    }

    for (ov::Output<ov::Node>& output : model->outputs()) {
        outputNames.push_back(output.get_any_name());
    }
}

std::unique_ptr<ResultBase> KeypointDetectionModel::postprocess(InferenceResult& infResult) {
    KeypointDetectionResult* result = new KeypointDetectionResult(infResult.frameId, infResult.metaData);
    return std::unique_ptr<ResultBase>(result);
}


std::unique_ptr<KeypointDetectionResult>
KeypointDetectionModel::infer(const ImageInputData& inputData) {
    auto result = ImageModel::inferImage(inputData);
    return std::unique_ptr<KeypointDetectionResult>(static_cast<KeypointDetectionResult*>(result.release()));
}

std::vector<std::unique_ptr<KeypointDetectionResult>> KeypointDetectionModel::inferBatch(const std::vector<ImageInputData>& inputImgs) {
    auto results = ImageModel::inferBatchImage(inputImgs);
    std::vector<std::unique_ptr<KeypointDetectionResult>> kpDetResults;
    kpDetResults.reserve(results.size());
    for (auto& result : results) {
        kpDetResults.emplace_back(static_cast<KeypointDetectionResult*>(result.release()));
    }
    return kpDetResults;
}
