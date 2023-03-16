/*
// Copyright (C) 2020-2023 Intel Corporation
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

#pragma once
#include <stddef.h>

#include <string>
#include <vector>

#include "models/detection_model.h"

struct DetectionResult;
struct ImageInputData;
struct InferenceAdatper;

class DetectionModelExt : public DetectionModel {
public:
    DetectionModelExt(const std::string& modelFile,
                   float confidenceThreshold,
                   const std::string& resize_type,
                   bool useAutoResize,
                   const std::vector<std::string>& labels,
                   float boxIOUThreshold,
                   const std::string& layout = "")
        : DetectionModel(modelFile, confidenceThreshold, resize_type, useAutoResize, labels, layout),
          boxIOUThreshold(boxIOUThreshold) {}

    DetectionModelExt(std::shared_ptr<ov::Model>& model, const ov::AnyMap& configuration)
        : DetectionModel(model, configuration) {
        auto iou_t_iter = configuration.find("iou_t");
        if (iou_t_iter != configuration.end()) {
            boxIOUThreshold = iou_t_iter->second.as<bool>();
        }
    }

    using DetectionModel::DetectionModel;

protected:
    float boxIOUThreshold = 0.5f;
};
