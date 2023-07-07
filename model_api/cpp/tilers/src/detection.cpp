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

#include <tilers/detection.h>
#include <models/results.h>
#include <utils/nms.hpp>

DetectionTiler::DetectionTiler(std::unique_ptr<ModelBase> _model, const ov::AnyMap& configuration) :
    TilerBase(std::move(_model), configuration) {

    auto ov_model = model->getModel();

    auto max_pred_iter = configuration.find("max_pred_number");
    if (max_pred_iter == configuration.end()) {
        if (ov_model->has_rt_info("model_info", "max_pred_number")) {
            max_pred_number = ov_model->get_rt_info<size_t>("model_info", "max_pred_number");
        }
    } else {
        max_pred_number = max_pred_iter->second.as<size_t>();
    }
}


std::unique_ptr<ResultBase> DetectionTiler::postprocess_tile(std::unique_ptr<ResultBase> tile_result, const cv::Rect& coord) {
    DetectionResult* det_res = static_cast<DetectionResult*>(tile_result.get());
    for (auto& det : det_res->objects) {
        det.x += coord.x;
        det.y += coord.y;
    }

    return tile_result;
}

std::unique_ptr<ResultBase> DetectionTiler::merge_results(const std::vector<std::unique_ptr<ResultBase>>& tiles_results, const cv::Size& image_size, const std::vector<cv::Rect>& tile_coords) {
    DetectionResult* result = new DetectionResult();
    auto retVal = std::unique_ptr<ResultBase>(result);

    std::vector<AnchorLabeled> all_detections;
    std::vector<DetectedObject*> all_detections_ptrs;
    std::vector<float> all_scores;

    for (const auto& result : tiles_results) {
        DetectionResult* det_res = static_cast<DetectionResult*>(result.get());
        for (auto& det : det_res->objects) {
            all_detections.emplace_back(det.x, det.y, det.x + det.width, det.y + det.height, det.labelID);
            all_scores.push_back(det.confidence);
            all_detections_ptrs.push_back(&det);
        }
    }

    auto keep_idx = multiclass_nms(all_detections, all_scores, 0.45f, false, 200);

    result->objects.reserve(keep_idx.size());
    for (auto idx : keep_idx) {
        result->objects.push_back(*all_detections_ptrs[idx]);
    }

    result->feature_vector = ov::Tensor();
    if (tiles_results.size()) {
        DetectionResult* det_res = static_cast<DetectionResult*>(tiles_results.begin()->get());
        result->feature_vector = ov::Tensor(det_res->feature_vector.get_element_type(), det_res->feature_vector.get_shape());
    }

    float* feature_ptr = result->feature_vector.data<float>();
    size_t feature_size = result->feature_vector.get_size();

    std::fill(feature_ptr, feature_ptr + feature_size, 0.f);

    for (const auto& result : tiles_results) {
        DetectionResult* det_res = static_cast<DetectionResult*>(result.get());
        const float* current_feature_ptr = det_res->feature_vector.data<float>();

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

cv::Mat wrap_to_mat(ov::Tensor& t, size_t shape_shift, size_t class_idx) {
    void* t_ptr;
    int ocv_dtype;

    if (t.get_element_type().get_type_name() == "u8") {
        t_ptr = t.data<unsigned char>();
        ocv_dtype = CV_8U;
    }
    else if (t.get_element_type().get_type_name() == "f32") {
        t_ptr = t.data<float>();
        ocv_dtype = CV_32F;
    }

    t_ptr += class_idx * t.get_strides()[shape_shift];
    /*
    std::vector<int> cv_shape;
    cv_shape.reserve(t.get_shape().size());
    for (auto s : t.get_shape()) {
        cv_shape.push_back(s);
    }

    std::vector<size_t> cv_strides;
    cv_strides.reserve(t.get_strides().size());
    for (auto s : t.get_strides()) {
        cv_strides.push_back(s);
    }
    */

    //return cv::Mat(2, cv_shape.data(), ocv_dtype, t_ptr, cv_strides.data());
    //	Mat (Size size, int type, void *data, size_t step=AUTO_STEP)
    return cv::Mat(cv::Size(t.get_shape()[shape_shift + 2], t.get_shape()[shape_shift + 1]), ocv_dtype, t_ptr, t.get_strides()[shape_shift + 1]);
}

ov::Tensor DetectionTiler::merge_saliency_maps(const std::vector<std::unique_ptr<ResultBase>>& tiles_results, const cv::Size& image_size, const std::vector<cv::Rect>& tile_coords) {
    std::vector<ov::Tensor> all_saliecy_maps;
    all_saliecy_maps.reserve(tiles_results.size());
    for (const auto& result : tiles_results) {
        DetectionResult* det_res = static_cast<DetectionResult*>(result.get());
        all_saliecy_maps.push_back(det_res->saliency_map);
    }

    ov::Tensor image_saliency_map;
    if (all_saliecy_maps.size()) {
        image_saliency_map = all_saliecy_maps[0];
    }

    if (image_saliency_map.get_size() == 1) {
        return image_saliency_map;
    }

    size_t shape_shift = (image_saliency_map.get_shape().size() > 3) ? 1 : 0;
    size_t num_classes = image_saliency_map.get_shape()[shape_shift];
    size_t map_h = image_saliency_map.get_shape()[shape_shift + 1];
    size_t map_w = image_saliency_map.get_shape()[shape_shift + 2];

    float ratio_h = static_cast<float>(map_h) / tile_size;
    float ratio_w = static_cast<float>(map_w) / tile_size;

    size_t image_map_h = static_cast<size_t>(image_size.height * ratio_h);
    size_t image_map_w = static_cast<size_t>(image_size.width * ratio_w);

    ov::Tensor merged_map;
    if (shape_shift) {
        merged_map = ov::Tensor(image_saliency_map.get_element_type(), {1, num_classes, image_map_h, image_map_w});
    }
    else {
        merged_map = ov::Tensor(image_saliency_map.get_element_type(), {num_classes, image_map_h, image_map_w});
    }

    for (size_t i = 1; i < all_saliecy_maps.size(); ++i) {
        for (size_t class_idx = 0; class_idx < num_classes; ++class_idx) {
            auto current_cls_map_mat = wrap_to_mat(all_saliecy_maps[i], shape_shift, class_idx);
            cv::Rect2i map_location(tile_coords[i].x * ratio_w, tile_coords[i].y * ratio_h,
                                  tile_coords[i].width * ratio_w, tile_coords[i].height * ratio_h);

            if (current_cls_map_mat.rows > map_location.height && map_location.height > 0 && current_cls_map_mat.cols > map_location.width && map_location.width > 0) {
                cv::resize(current_cls_map_mat, current_cls_map_mat, cv::Size(map_location.width, map_location.height));
            }

        }
    }


    return merged_map;
}
