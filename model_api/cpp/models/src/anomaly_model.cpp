/*
  Copyright (C) 2020-2023 Intel Corporation

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/

#include "models/anomaly_model.h"
#include "models/image_model.h"
#include "models/input_data.h"
#include "models/internal_model_data.h"
#include "models/results.h"
#include "utils/slog.hpp"
#include <memory>
#include <openvino/core/any.hpp>
#include <openvino/core/model.hpp>
#include <ostream>

std::string AnomalyModel::ModelType = "AnomalyDetection";

AnomalyModel::AnomalyModel(std::shared_ptr<ov::Model> &model,
                           const ov::AnyMap &configuration)
    : ImageModel(model, configuration) {
  auto imageThreshold_iter = configuration.find("image_threshold");
  if (imageThreshold_iter == configuration.end()) {
    if (model->has_rt_info("model_info", "image_threshold")) {
      imageThreshold =
          model->get_rt_info<float>("model_info", "image_threshold");
    }
  } else {
    imageThreshold = imageThreshold_iter->second.as<float>();
  }

  auto pixelThreshold_iter = configuration.find("pixel_threshold");
  if (pixelThreshold_iter == configuration.end()) {
    if (model->has_rt_info("model_info", "pixel_threshold")) {
      pixelThreshold =
          model->get_rt_info<float>("model_info", "pixel_threshold");
    }
  } else {
    pixelThreshold = pixelThreshold_iter->second.as<float>();
  }

  auto max_iter = configuration.find("max");
  if (max_iter == configuration.end()) {
    if (model->has_rt_info("model_info", "max")) {
      max = model->get_rt_info<float>("model_info", "max");
    }
  } else {
    max = max_iter->second.as<float>();
  }

  auto min_iter = configuration.find("min");
  if (min_iter == configuration.end()) {
    if (model->has_rt_info("model_info", "min")) {
      min = model->get_rt_info<float>("model_info", "min");
    }
  } else {
    min = min_iter->second.as<float>();
  }
}

AnomalyModel::AnomalyModel(std::shared_ptr<InferenceAdapter> &adapter)
    : ImageModel(adapter) {
  const ov::AnyMap &config = adapter->getModelConfig();

  auto imageThreshold_iter = config.find("image_threshold");
  if (imageThreshold_iter != config.end()) {
    imageThreshold = imageThreshold_iter->second.as<float>();
  }

  auto pixelThreshold_iter = config.find("pixel_threshold");
  if (pixelThreshold_iter != config.end()) {
    pixelThreshold = pixelThreshold_iter->second.as<float>();
  }

  auto max_iter = config.find("max");
  if (max_iter != config.end()) {
    max = max_iter->second.as<float>();
  }

  auto min_iter = config.find("min");
  if (min_iter != config.end()) {
    min = min_iter->second.as<float>();
  }
}

std::shared_ptr<InternalModelData>
AnomalyModel::preprocess(const InputData &inputData, InferenceInput &input) {

  // auto processedData = ImageModel::preprocess(inputData, input);
  // return processedData;
}

std::unique_ptr<AnomalyResult>
AnomalyModel::infer(const ImageInputData &inputData) {
  // return AnomalyResult{};
}

std::unique_ptr<ResultBase>
AnomalyModel::postprocess(InferenceResult &infResult) {
  return std::make_unique<ResultBase>(infResult);
}

std::unique_ptr<AnomalyModel> AnomalyModel::create_model(
    const std::string &modelFile, const ov::AnyMap &configuration,
    std::string model_type, bool preload, const std::string &device) {
  auto core = ov::Core();
  std::shared_ptr<ov::Model> model = core.read_model(modelFile);
  if (model_type.empty()) {
    try {
      if (model->has_rt_info("model_info", "model_type")) {
        model_type =
            model->get_rt_info<std::string>("model_info", "model_type");
      }
    } catch (const std::exception &) {
      slog::warn << "Model type is not specified in the rt_info, use default "
                    "model type: "
                 << model_type << slog::endl;
    }
  }
  std::unique_ptr<AnomalyModel> anomalyModel;
  if (model_type == AnomalyModel::ModelType)
    anomalyModel =
        std::unique_ptr<AnomalyModel>(new AnomalyModel(model, configuration));
  else
    throw std::runtime_error("Incorrect or unsupported model_type is provided "
                             "in the model_info section: " +
                             model_type);

  anomalyModel->prepare();
  if (preload) {
    anomalyModel->load(core, device);
  }
  return anomalyModel;
}

std::unique_ptr<AnomalyModel>
AnomalyModel::create_model(std::shared_ptr<InferenceAdapter> &adapter) {
  const ov::AnyMap &configuration = adapter->getModelConfig();
  auto model_type_iter = configuration.find("model_type");
  std::string model_type = AnomalyModel::ModelType;
  if (model_type_iter != configuration.end()) {
    model_type = model_type_iter->second.as<std::string>();
  }

  if (model_type != AnomalyModel::ModelType) {
    throw std::runtime_error(
        "Incorrect or unsupported model_type is provided: " + model_type);
  }
  if (model_type != AnomalyModel::ModelType) {
    throw std::runtime_error(
        "Incorrect or unsupported model_type is provided: " + model_type);
  }
  std::unique_ptr<AnomalyModel> anomalyModel{new AnomalyModel(adapter)};
  return anomalyModel;
}

std::ostream &operator<<(std::ostream &os,
                         std::unique_ptr<AnomalyModel> &model) {
  os << "AnomalyModel: " << model->task << ", Image threshold"
     << model->imageThreshold << ", Pixel threshold" << model->pixelThreshold
     << ", Max" << model->max << ", Min" << model->min;
  return os;
}

void AnomalyModel::prepareInputsOutputs(std::shared_ptr<ov::Model> &model) {
  const auto &input = model->input();
  inputNames.push_back(input.get_any_name());

  const ov::Shape &inputShape = input.get_partial_shape().get_max_shape();
  const ov::Layout &inputLayout = getInputLayout(input);

  if (!embedded_processing) {
    model = ImageModel::embedProcessing(
        model, inputNames[0], inputLayout, resizeMode, interpolationMode,
        ov::Shape{inputShape[ov::layout::width_idx(inputLayout)],
                  inputShape[ov::layout::height_idx(inputLayout)]},
        pad_value, reverse_input_channels, {}, scale_values);

    ov::preprocess::PrePostProcessor ppp =
        ov::preprocess::PrePostProcessor(model);
    model = ppp.build();
  }
}

void AnomalyModel::updateModelInfo() { ImageModel::updateModelInfo(); }