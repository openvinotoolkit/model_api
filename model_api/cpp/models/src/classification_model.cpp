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

#include "models/classification_model.h"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <openvino/op/softmax.hpp>
#include <openvino/op/topk.hpp>
#include <openvino/openvino.hpp>

#include <utils/slog.hpp>

#include "models/results.h"
#include "models/input_data.h"

std::string ClassificationModel::ModelType = "Classification";

ClassificationModel::ClassificationModel(std::shared_ptr<ov::Model>& model, const ov::AnyMap& configuration)
    : ImageModel(model, configuration) {
    auto topk_iter = configuration.find("topk");
    if (topk_iter == configuration.end()) {
        if (model->has_rt_info("model_info", "topk")) {
            topk = stoi(model->get_rt_info<std::string>("model_info", "topk"));
        }
    } else {
        topk = topk_iter->second.as<size_t>();
    }
}

ClassificationModel::ClassificationModel(std::shared_ptr<InferenceAdapter>& adapter)
    : ImageModel(adapter) {
    auto configuration = adapter->getModelConfig();
    auto topk_iter = configuration.find("topk");
    if (topk_iter != configuration.end()) {
        topk = topk_iter->second.as<size_t>();
    }
}

void ClassificationModel::updateModelInfo() {
    ImageModel::updateModelInfo();

    model->set_rt_info(ClassificationModel::ModelType, "model_info", "model_type");
    model->set_rt_info(topk, "model_info", "topk");
}

std::unique_ptr<ClassificationModel> ClassificationModel::create_model(const std::string& modelFile, const ov::AnyMap& configuration, bool preload) {
    auto core = ov::Core();
    std::shared_ptr<ov::Model> model = core.read_model(modelFile);
    
    // Check model_type in the rt_info, ignore configuration
    std::string model_type = ClassificationModel::ModelType;
    try {
        if (model->has_rt_info("model_info", "model_type")) {
            model_type = model->get_rt_info<std::string>("model_info", "model_type");
        }
    } catch (const std::exception& e) {
        slog::warn << "Model type is not specified in the rt_info, use default model type: " << model_type << slog::endl;
    }
    
    if (model_type != ClassificationModel::ModelType) {
        throw ov::Exception("Incorrect or unsupported model_type is provided in the model_info section: " + model_type);
    }

    std::unique_ptr<ClassificationModel> classifier{new ClassificationModel(model, configuration)};
    classifier->prepare();
    if (preload) {
        classifier->load(core);
    }
    return classifier;
}

std::unique_ptr<ClassificationModel> ClassificationModel::create_model(std::shared_ptr<InferenceAdapter>& adapter) {
    auto configuration = adapter->getModelConfig();
    auto model_type_iter = configuration.find("model_type");
    std::string model_type = ClassificationModel::ModelType;
    if (model_type_iter != configuration.end()) {
        model_type = model_type_iter->second.as<std::string>();
    }

    if (model_type != ClassificationModel::ModelType) {
        throw ov::Exception("Incorrect or unsupported model_type is provided: " + model_type);
    }

    std::unique_ptr<ClassificationModel> classifier{new ClassificationModel(adapter)};
    return classifier;
}

static void softmax(std::vector<float>& input) {
	size_t i;
	float min_val, sum, shift;
    size_t size = input.size();

	min_val = -INFINITY;
	for (i = 0; i < size; ++i) {
		if (min_val < input[i]) {
			min_val = input[i];
		}
	}

	sum = 0.0;
	for (i = 0; i < size; ++i) {
		sum += exp(input[i] - min_val);
	}
    // Shift is required to prevent overflow
	shift = min_val + log(sum);
	for (i = 0; i < size; ++i) {
		input[i] = exp(input[i] - shift);
	}
}

std::unique_ptr<ResultBase> ClassificationModel::postprocess(InferenceResult& infResult) {
    ClassificationResult* result = new ClassificationResult(infResult.frameId, infResult.metaData);
    auto retVal = std::unique_ptr<ResultBase>(result);
    std::vector<int> indices;
    std::vector<float> scores;
    
    if (embedded_processing)
    {
        const ov::Tensor& indicesTensor = infResult.outputsData.find(outputNames[0])->second;
        const int* indicesPtr = indicesTensor.data<int>();
        std::copy(indicesPtr, indicesPtr + topk, std::back_inserter(indices));

        const ov::Tensor& scoresTensor = infResult.outputsData.find(outputNames[1])->second;
        const float* scoresPtr = scoresTensor.data<float>();
        std::copy(scoresPtr, scoresPtr + topk, std::back_inserter(scores));
    } else {
        const ov::Tensor& logitTensor = infResult.outputsData.find(outputNames[0])->second;
        const float* logits = logitTensor.data<float>();
        size_t num_classes = logitTensor.get_size();
        std::vector<float> buffer;
        std::copy(logits, logits + num_classes, std::back_inserter(buffer));

        // apply softmax
        softmax(buffer);
        // sort in the reverse order
        std::vector<size_t> idx(buffer.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::stable_sort(idx.begin(), idx.end(), 
                        [&buffer](size_t i1, size_t i2) {return buffer[i1] > buffer[i2];});
        std::copy(std::begin(idx), std::begin(idx) + topk, std::back_inserter(indices));
        scores.reserve(indices.size());
        for (auto index : indices)
        {
            scores.push_back(buffer[index]);
        }
    }

    result->topLabels.reserve(topk);
    for (size_t i = 0; i < topk; ++i) {
        int ind = indices[i];
        if (ind < 0 || ind >= static_cast<int>(labels.size())) {
            throw std::runtime_error("Invalid index for the class label is found during postprocessing");
        }
        result->topLabels.emplace_back(ind, labels[ind], scores[i]);
    }

    return retVal;
}

void ClassificationModel::prepareInputsOutputs(std::shared_ptr<ov::Model>& model) {
    // --------------------------- Configure input & output -------------------------------------------------
    // --------------------------- Prepare input  ------------------------------------------------------
    if (model->inputs().size() != 1) {
        throw std::logic_error("Classification model wrapper supports topologies with only 1 input");
    }
    const auto& input = model->input();
    inputNames.push_back(input.get_any_name());

    outputNames.push_back("indices");
    outputNames.push_back("scores");

    // Skip next steps if pre/postprocessing was embedded previously
    if (embedded_processing) {
        slog::info << "Skip pre/postprocessing embedding for the model" << slog::endl;
        return;
    }

    const ov::Shape& inputShape = input.get_partial_shape().get_max_shape();
    const ov::Layout& inputLayout = getInputLayout(input);

    auto graphResizeMode = resizeMode;
    if (!useAutoResize) {
        graphResizeMode = NO_RESIZE;
    }

    model = ImageModel::embedProcessing(model,
                                        inputNames[0],
                                        inputLayout,
                                        graphResizeMode,
                                        interpolationMode,
                                        ov::Shape{inputShape[3], inputShape[2]});                                       

    // --------------------------- Adding softmax and topK output  ---------------------------
    auto nodes = model->get_ops();
    auto softmaxNodeIt = std::find_if(std::begin(nodes), std::end(nodes), [](const std::shared_ptr<ov::Node>& op) {
        return std::string(op->get_type_name()) == "Softmax"; // TODO: it will not work for Vision Transformers, for example
    });

    std::shared_ptr<ov::Node> softmaxNode;
    if (softmaxNodeIt == nodes.end()) {
        auto logitsNode = model->get_output_op(0)->input(0).get_source_output().get_node();
        softmaxNode = std::make_shared<ov::op::v1::Softmax>(logitsNode->output(0), 1);
    } else {
        softmaxNode = *softmaxNodeIt;
    }
    const auto k = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<size_t>{topk});
    std::shared_ptr<ov::Node> topkNode = std::make_shared<ov::op::v3::TopK>(softmaxNode,
                                                                            k,
                                                                            1,
                                                                            ov::op::v3::TopK::Mode::MAX,
                                                                            ov::op::v3::TopK::SortType::SORT_VALUES);

    auto indices = std::make_shared<ov::op::v0::Result>(topkNode->output(0));
    auto scores = std::make_shared<ov::op::v0::Result>(topkNode->output(1));
    ov::ResultVector res({scores, indices});
    model = std::make_shared<ov::Model>(res, model->get_parameters(), "classification");

    // manually set output tensors name for created topK node
    model->outputs()[0].set_names({"indices"});
    model->outputs()[1].set_names({"scores"});

    // set output precisions
    auto ppp = ov::preprocess::PrePostProcessor(model);
    ppp.output("indices").tensor().set_element_type(ov::element::i32);
    ppp.output("scores").tensor().set_element_type(ov::element::f32);
    model = ppp.build();
    embedded_processing = true;
}

std::unique_ptr<ClassificationResult> ClassificationModel::infer(const ImageInputData& inputData) {
    auto result = ModelBase::infer(static_cast<const InputData&>(inputData));
    return std::unique_ptr<ClassificationResult>(static_cast<ClassificationResult*>(result.release()));
}
