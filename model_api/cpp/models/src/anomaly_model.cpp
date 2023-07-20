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

  auto mean_values_iter = configuration.find("mean_values");
  if (mean_values_iter == configuration.end()) {
    if (model->has_rt_info("model_info", "mean_values")) {
      mean_values =
          model->get_rt_info<std::vector<float>>("model_info", "mean_values");
    }
  } else {
    mean_values = mean_values_iter->second.as<std::vector<float>>();
  }

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

  auto task_iter = configuration.find("task");
  if (task_iter == configuration.end()){
    if (model->has_rt_info("model_info", "task")) {
      task = model->get_rt_info<std::string>("model_info", "task");
    }
  } else {
    task = task_iter->second.as<std::string>();
  }
}

AnomalyModel::AnomalyModel(std::shared_ptr<InferenceAdapter> &adapter)
    : ImageModel(adapter) {
  const ov::AnyMap &config = adapter->getModelConfig();

  auto mean_values_iter = config.find("mean_values");
  if (mean_values_iter != config.end()) {
    mean_values = mean_values_iter->second.as<std::vector<float>>();
  }

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

  auto task_iter = config.find("task");
  if (task_iter != config.end()) {
    task = task_iter->second.as<std::string>();
  }
}

std::unique_ptr<AnomalyResult>
AnomalyModel::infer(const ImageInputData &inputData) {
  auto result = ModelBase::infer(static_cast<const InputData &>(inputData));

  return std::unique_ptr<AnomalyResult>(
      static_cast<AnomalyResult *>(result.release()));
}

std::unique_ptr<ResultBase>
AnomalyModel::postprocess(InferenceResult &infResult) {
  ov::Tensor predictions = infResult.outputsData[outputNames[0]];
  const auto &inputImgSize =
      infResult.internalModelData->asRef<InternalImageModelData>();

  double pred_score;
  std::string pred_label;
  cv::Mat anomaly_map;
  cv::Mat pred_mask;
  std::vector<std::vector<int>> pred_boxes;
  if (predictions.get_shape().size() == 1) {
    pred_score = predictions.data<float>()[0];
  } else {
    anomaly_map = cv::Mat(imageShape[0], imageShape[1], CV_32FC1,
                          predictions.data<float>());
    // find the max predicted score
    cv::minMaxLoc(anomaly_map, NULL, &pred_score); 
    // pred_score should be 56.030
    // minvalue 4.9
    // min value and max values are wrong
  }
  pred_label = labels[pred_score > imageThreshold ? 1 : 0];

  if (task == "segmentation" || task == "detection") {
    pred_mask = anomaly_map>=pixelThreshold;
    pred_mask.convertTo(pred_mask, CV_8UC1, 255);
    cv::imwrite("/home/ashwin/projects/model_api/anomaly/build/"
                "pred_mask_intermiediate.png",
                pred_mask);
    anomaly_map = normalize(anomaly_map, pixelThreshold);
  }
  pred_score = normalize(pred_score, imageThreshold);
  if (!anomaly_map.empty()) {
    cv::resize(
        anomaly_map, anomaly_map,
        cv::Size{inputImgSize.inputImgWidth, inputImgSize.inputImgHeight});
    cv::resize(
        pred_mask, pred_mask,
        cv::Size{inputImgSize.inputImgWidth, inputImgSize.inputImgHeight});
  }
  if (task == "detection") {
    pred_boxes = getBoxes(pred_mask);
  }

  AnomalyResult *result =
      new AnomalyResult(infResult.frameId, infResult.metaData);
  result->anomaly_map = anomaly_map;
  result->pred_score = pred_score;
  result->pred_label = pred_label;
  result->pred_mask = pred_mask;
  result->pred_boxes = pred_boxes;
  return std::unique_ptr<ResultBase>(result);
}

cv::Mat AnomalyModel::normalize(cv::Mat &tensor, float threshold) {
  cv::Mat normalized = ((tensor - threshold) / (max - min)) + 0.5f;
  for (auto row = 0; row < normalized.rows; row++) {
    for (auto col = 0; col < normalized.cols; col++) {
      normalized.at<float>(row, col) =
          std::min(std::max(normalized.at<float>(row, col), 0.f), 1.f);
    }
  }
  return normalized;
}

double AnomalyModel::normalize(double &value, float threshold) {
  double normalized = ((value - threshold) / (max - min))+0.5f;
  return std::min(std::max(normalized, 0.), 1.);
}

std::vector<std::vector<int>> AnomalyModel::getBoxes(cv::Mat &mask) {
  std::vector<std::vector<int>> boxes;
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  for (auto &contour : contours) {
    std::vector<int> box;
    cv::Rect rect = cv::boundingRect(contour);
    box.push_back(rect.x);
    box.push_back(rect.y);
    box.push_back(rect.x + rect.width);
    box.push_back(rect.y + rect.height);
    boxes.push_back(box);
  }
  return boxes;
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
  model->set_rt_info(std::vector<std::string>{"Normal", "Anomalous"},
                     "model_info", "labels");
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
  os << "AnomalyModel: " << model->task
     << ", Image threshold: " << model->imageThreshold
     << ", Pixel threshold: " << model->pixelThreshold
     << ", Max: " << model->max << ", Min: " << model->min << std::endl;
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
        pad_value, reverse_input_channels, mean_values, scale_values,
        typeid(float));
    embedded_processing = true;
  }

  const auto &outputs = model->outputs();
  for (const auto &output : model->outputs()) {
    outputNames.push_back(output.get_any_name());
  }
}

void AnomalyModel::updateModelInfo() { ImageModel::updateModelInfo(); }

std::shared_ptr<InternalModelData>
AnomalyModel::preprocess(const InputData &inputData, InferenceInput &input) {
  const auto &origImg = inputData.asRef<ImageInputData>().inputImage;
  auto img = inputTransform(origImg);
  input.emplace(inputNames[0], wrapMat2Tensor(img));
  return std::make_shared<InternalImageModelData>(origImg.cols, origImg.rows);
}