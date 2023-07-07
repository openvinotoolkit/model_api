/*
// Copyright (C) 2023 Intel Corporation
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

#include <algorithm>
#include <vector>
#include <opencv2/core.hpp>

#include <tilers/instance_segmentation.h>
#include <models/results.h>
#include <utils/nms.hpp>

InstanceSegmentationTiler::InstanceSegmentationTiler(std::unique_ptr<ModelBase> _model, const ov::AnyMap& configuration) :
    DetectionTiler(std::move(_model), configuration) {}


std::unique_ptr<ResultBase> InstanceSegmentationTiler::postprocess_tile(std::unique_ptr<ResultBase> tile_result, const cv::Rect& coord) {
    auto* iseg_res = static_cast<InstanceSegmentationResult*>(tile_result.get());
    for (auto& det : iseg_res->segmentedObjects) {
        det.x += coord.x;
        det.y += coord.y;
    }

    return tile_result;
}

std::unique_ptr<ResultBase> InstanceSegmentationTiler::merge_results(const std::vector<std::unique_ptr<ResultBase>>& tiles_results, const cv::Size& image_size, const std::vector<cv::Rect>& tile_coords) {
    auto* result = new InstanceSegmentationResult();
    auto retVal = std::unique_ptr<ResultBase>(result);

    std::vector<AnchorLabeled> all_detections;
    std::vector<SegmentedObject*> all_detections_ptrs;
    std::vector<float> all_scores;

    for (const auto& result : tiles_results) {
        auto* iseg_res = static_cast<InstanceSegmentationResult*>(result.get());
        for (auto& det : iseg_res->segmentedObjects) {
            all_detections.emplace_back(det.x, det.y, det.x + det.width, det.y + det.height, det.labelID);
            all_scores.push_back(det.confidence);
            all_detections_ptrs.push_back(&det);
        }
    }

    auto keep_idx = multiclass_nms(all_detections, all_scores, 0.45f, false, 200);

    result->segmentedObjects.reserve(keep_idx.size());
    for (auto idx : keep_idx) {
        result->segmentedObjects.push_back(*all_detections_ptrs[idx]);
    }

    result->feature_vector = ov::Tensor();
    if (tiles_results.size()) {
        auto* iseg_res = static_cast<InstanceSegmentationResult*>(tiles_results.begin()->get());
        result->feature_vector = ov::Tensor(iseg_res->feature_vector.get_element_type(), iseg_res->feature_vector.get_shape());
    }

    float* feature_ptr = result->feature_vector.data<float>();
    size_t feature_size = result->feature_vector.get_size();

    std::fill(feature_ptr, feature_ptr + feature_size, 0.f);

    for (const auto& result : tiles_results) {
        auto* iseg_res = static_cast<InstanceSegmentationResult*>(result.get());
        const float* current_feature_ptr = iseg_res->feature_vector.data<float>();

        for (size_t i = 0; i < feature_size; ++i) {
            feature_ptr[i] += current_feature_ptr[i];
        }
    }

    for (size_t i = 0; i < feature_size; ++i) {
        feature_ptr[i] /= tiles_results.size();
    }

    result->saliency_map = merge_saliency_maps(tiles_results, image_size, tile_coords);

    return retVal;
}

std::vector<cv::Mat_<std::uint8_t>> InstanceSegmentationTiler::merge_saliency_maps(const std::vector<std::unique_ptr<ResultBase>>& tile_results, const cv::Size& image_size, const std::vector<cv::Rect>& tile_coords) {

}
