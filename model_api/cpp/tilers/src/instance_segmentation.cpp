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
#include "utils/common.hpp"

cv::Rect expand_box(const cv::Rect2f& box, float scale) {
    float w_half = box.width * 0.5f * scale,
        h_half = box.height * 0.5f * scale;
    const cv::Point2f& center = (box.tl() + box.br()) * 0.5f;
    return {cv::Point(int(center.x - w_half), int(center.y - h_half)), cv::Point(int(center.x + w_half), int(center.y + h_half))};
}

cv::Mat segm_postprocess(const SegmentedObject& box, const cv::Mat& unpadded, int im_h, int im_w) {
    // Add zero border to prevent upsampling artifacts on segment borders.
    cv::Mat raw_cls_mask;
    cv::copyMakeBorder(unpadded, raw_cls_mask, 1, 1, 1, 1, cv::BORDER_CONSTANT, {0});
    cv::Rect extended_box = expand_box(box, float(raw_cls_mask.cols) / (raw_cls_mask.cols - 2));

    int w = std::max(extended_box.width + 1, 1);
    int h = std::max(extended_box.height + 1, 1);
    int x0 = clamp(extended_box.x, 0, im_w);
    int y0 = clamp(extended_box.y, 0, im_h);
    int x1 = clamp(extended_box.x + extended_box.width + 1, 0, im_w);
    int y1 = clamp(extended_box.y + extended_box.height + 1, 0, im_h);

    cv::Mat resized;
    cv::resize(raw_cls_mask, resized, {w, h});
    cv::Mat im_mask(cv::Size{im_w, im_h}, CV_8UC1, cv::Scalar{0});
    im_mask(cv::Rect{x0, y0, x1-x0, y1-y0}).setTo(1, resized({cv::Point(x0-extended_box.x, y0-extended_box.y), cv::Point(x1-extended_box.x, y1-extended_box.y)}) > 0.5f);
    return im_mask;
}

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
        all_detections_ptrs[idx]->mask = segm_postprocess(*all_detections_ptrs[idx], all_detections_ptrs[idx]->mask,
                                    image_size.height, image_size.width);
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

std::vector<cv::Mat_<std::uint8_t>> InstanceSegmentationTiler::merge_saliency_maps(const std::vector<std::unique_ptr<ResultBase>>& tiles_results,
                                                                                   const cv::Size& image_size, const std::vector<cv::Rect>& tile_coords) {
    std::vector<std::vector<cv::Mat_<std::uint8_t>>> all_saliecy_maps;
    all_saliecy_maps.reserve(tiles_results.size());
    for (const auto& result : tiles_results) {
        auto det_res = static_cast<InstanceSegmentationResult*>(result.get());
        all_saliecy_maps.push_back(det_res->saliency_map);
    }

    std::vector<cv::Mat_<std::uint8_t>> image_saliency_map;
    if (all_saliecy_maps.size()) {
        image_saliency_map = all_saliecy_maps[0];
    }

    if (image_saliency_map.empty()) {
        return image_saliency_map;
    }

    size_t num_classes = image_saliency_map.size();
    std::vector<cv::Mat_<std::uint8_t>> merged_map(num_classes);
    for (auto& map : merged_map) {
        map = cv::Mat_<std::uint8_t>(image_size);
        map = 0;
    }

    for (size_t i = 1; i < all_saliecy_maps.size(); ++i) {
        for (size_t class_idx = 0; class_idx < num_classes; ++class_idx) {
            auto current_cls_map_mat = all_saliecy_maps[i][class_idx];
            if (current_cls_map_mat.empty()) {
                continue;
            }
            const auto& tile = tile_coords[i];
            cv::Mat tile_map;
            cv::resize(current_cls_map_mat, tile_map, tile.size());
            auto tile_map_merged = cv::Mat(merged_map[class_idx], tile);
            cv::Mat(cv::max(tile_map, tile_map_merged)).copyTo(tile_map_merged);
        }
    }

    for (size_t class_idx = 0; class_idx < num_classes; ++class_idx) {
        auto image_map_cls = image_saliency_map[class_idx];
        if (image_map_cls.empty()) {
            continue;
        }
        cv::resize(image_map_cls, image_map_cls, image_size);
        cv::addWeighted(merged_map[class_idx], 1.0, image_map_cls, 0.5, 0., merged_map[class_idx]);
    }

    return merged_map;
}
