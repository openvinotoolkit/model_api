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

#include <condition_variable>
#include <functional>
#include <queue>
#include <vector>

#include <openvino/openvino.hpp>


class AsyncInferQueue {
public:
    AsyncInferQueue() = default;
    AsyncInferQueue(ov::CompiledModel& model, size_t jobs);
    ~AsyncInferQueue();
    bool is_ready();
    size_t get_idle_request_id();
    void wait_all();
    void set_default_callbacks();
    void set_custom_callbacks(std::function<void(ov::InferRequest, const ov::AnyMap& callback_args)> f_callback);
    size_t size() const;
    void start_async(const ov::Tensor& input, std::shared_ptr<ov::AnyMap>& userdata);
    void start_async(const std::map<std::string, ov::Tensor>& input, std::shared_ptr<ov::AnyMap>& userdata);
    ov::InferRequest operator[](size_t i);

    // AsyncInferQueue is the owner of all requests. When AsyncInferQueue is destroyed,
    // all of requests are destroyed as well.
    std::vector<ov::InferRequest> m_requests;
    std::queue<size_t> m_idle_handles;
    std::vector<std::shared_ptr<ov::AnyMap>> m_user_ids;
    std::mutex m_mutex;
    std::condition_variable m_cv;
    std::queue<std::shared_ptr<std::exception>> m_errors;
};
