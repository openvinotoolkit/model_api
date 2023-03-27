/*
// Copyright (C) 2021-2023 Intel Corporation
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

#include "utils/image_utils.h"

#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset10.hpp>
#include <openvino/opsets/opset4.hpp>

using namespace ov;

cv::Mat resizeImageExt(const cv::Mat& mat, int width, int height, RESIZE_MODE resizeMode,
                       cv::InterpolationFlags interpolationMode, cv::Rect* roi, cv::Scalar BorderConstant) {
    if (width == mat.cols && height == mat.rows) {
        return mat;
    }

    cv::Mat dst;

    switch (resizeMode) {
    case RESIZE_FILL:
    {
        cv::resize(mat, dst, cv::Size(width, height), interpolationMode);
        if (roi) {
            *roi = cv::Rect(0, 0, width, height);
        }
        break;
    }
    case RESIZE_KEEP_ASPECT:
    case RESIZE_KEEP_ASPECT_LETTERBOX:
    {
        double scale = std::min(static_cast<double>(width) / mat.cols, static_cast<double>(height) / mat.rows);
        cv::Mat resizedImage;
        cv::resize(mat, resizedImage, {int(mat.cols * scale), int(mat.rows * scale)}, 0, 0, interpolationMode);

        int dx = resizeMode == RESIZE_KEEP_ASPECT ? 0 : (width - resizedImage.cols) / 2;
        int dy = resizeMode == RESIZE_KEEP_ASPECT ? 0 : (height - resizedImage.rows) / 2;

        cv::copyMakeBorder(resizedImage, dst, dy, height - resizedImage.rows - dy,
            dx, width - resizedImage.cols - dx, cv::BORDER_CONSTANT, BorderConstant);
        if (roi) {
            *roi = cv::Rect(dx, dy, resizedImage.cols, resizedImage.rows);
        }
        break;
    }
    case NO_RESIZE:
    {
        dst = mat;
        if (roi) {
            *roi = cv::Rect(0, 0, mat.cols, mat.rows);
        }
        break;
    }
    }
    return dst;
}

//ov::preprocess::PostProcessSteps::CustomPostprocessOp
static opset10::Interpolate::InterpolateMode ov2ovInterpolationMode(cv::InterpolationFlags interpolationMode) {
    switch (interpolationMode)
    {
        case cv::INTER_NEAREST:
            return opset10::Interpolate::InterpolateMode::NEAREST;
            break;
        case cv::INTER_LINEAR:
            return opset10::Interpolate::InterpolateMode::LINEAR;
        case cv::INTER_CUBIC:
            return opset10::Interpolate::InterpolateMode::CUBIC;
        default:
            break;
    }

    return opset10::Interpolate::InterpolateMode::LINEAR;
}

static Output<Node> resizeImageGraph(const ov::Output<ov::Node>& input,
                const ov::Shape& size, 
                bool keep_aspect_ratio = false,
                const cv::InterpolationFlags interpolationMode = cv::INTER_LINEAR) {
    const auto h_axis = 1;
    const auto w_axis = 2;

    auto mode = ov2ovInterpolationMode(interpolationMode);

    if (size.size() != 2) {
        throw std::logic_error("Size parameter should be 2-dimensional");
    }
    auto w = size.at(0);
    auto h = size.at(1);

    auto sizes = opset10::Constant::create(element::i64, Shape{2}, {h,w});
    auto axes = opset10::Constant::create(element::i64, Shape{2}, {h_axis, w_axis});

    if (!keep_aspect_ratio) {
        auto scales = opset10::Constant::create(element::f32, Shape{2}, {1.0f, 1.0f});
        opset10::Interpolate::InterpolateAttrs attrs;
        attrs.mode = mode;
        attrs.shape_calculation_mode = opset10::Interpolate::ShapeCalcMode::SIZES;

        return std::make_shared<opset10::Interpolate>(input, sizes, scales, axes, attrs);
    }
    
    auto image_shape = std::make_shared<opset10::ShapeOf>(input);
    auto iw = std::make_shared<opset10::Convert>(std::make_shared<opset10::Gather>(image_shape, 
                                                                                   opset10::Constant::create(element::i64, Shape{1}, {w_axis}), 
                                                                                   opset10::Constant::create(element::i64, Shape{1}, {0})),
                                                element::f32);
    auto ih = std::make_shared<opset10::Convert>(std::make_shared<opset10::Gather>(image_shape, 
                                                                                   opset10::Constant::create(element::i64, Shape{1}, {h_axis}), 
                                                                                   opset10::Constant::create(element::i64, Shape{1}, {0})),
                                                element::f32);

    auto w_ratio = std::make_shared<opset10::Divide>(opset10::Constant::create(element::f32, Shape{1}, {float(w)}), iw);
    auto h_ratio = std::make_shared<opset10::Divide>(opset10::Constant::create(element::f32, Shape{1}, {float(h)}), ih);
    auto scale = std::make_shared<opset10::Minimum>(w_ratio, h_ratio);

    auto scales = opset10::Constant::create(element::f32, Shape{2}, {1.0f, 1.0f});
    opset10::Interpolate::InterpolateAttrs attrs;
    attrs.mode = mode;
    attrs.shape_calculation_mode = opset10::Interpolate::ShapeCalcMode::SCALES;

    return std::make_shared<opset10::Interpolate>(input, sizes, scales, axes, attrs);
}


preprocess::PostProcessSteps::CustomPostprocessOp createResizeGraph(RESIZE_MODE resizeMode,
                                                                    const Shape& size, 
                                                                    const cv::InterpolationFlags interpolationMode) {
    if (resizeMode == RESIZE_FILL) {
        return [=](const Output<Node>& node) {
            return resizeImageGraph(node, size, false, interpolationMode);
        };
    }

    throw std::runtime_error("Unsupported resize mode: " + resizeMode);
}