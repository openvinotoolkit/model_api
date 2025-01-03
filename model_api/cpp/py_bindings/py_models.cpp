#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/operators.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include <opencv2/core/types.hpp>
#include <openvino/openvino.hpp>

#include "models/classification_model.h"
#include "models/results.h"

namespace nb = nanobind;

namespace {
cv::Mat wrap_np_mat(const nb::ndarray<>& input) {
    if (input.ndim() != 3 || input.shape(2) != 3 || input.dtype() != nb::dtype<uint8_t>()) {
        throw std::runtime_error("Input image should have HWC_8U layout");
    }

    int height = input.shape(0);
    int width = input.shape(1);

    return cv::Mat(height, width, CV_8UC3, input.data());
}

ov::Any py_object_to_any(const nb::object& py_obj, std::string property_name) {
    if (nb::isinstance<nb::str>(py_obj)) {
        return ov::Any(std::string(static_cast<nb::str>(py_obj).c_str()));
    } else if (nb::isinstance<nb::float_>(py_obj)) {
        return ov::Any(static_cast<double>(static_cast<nb::float_>(py_obj)));
    } else if (nb::isinstance<nb::int_>(py_obj)) {
        return ov::Any(static_cast<int>(static_cast<nb::int_>(py_obj)));
    } else {
        OPENVINO_THROW("Property \"" + property_name + "\" has unsupported type.");
    }
}

}  // namespace

NB_MODULE(py_model_api, m) {
    m.doc() = "Nanobind binding for OpenVINO Vision API library";
    nb::class_<ResultBase>(m, "ResultBase").def(nb::init<>());

    nb::class_<ClassificationResult::Classification>(m, "Classification")
        .def(nb::init<unsigned int, const std::string, float>())
        .def_rw("id", &ClassificationResult::Classification::id)
        .def_rw("label", &ClassificationResult::Classification::label)
        .def_rw("score", &ClassificationResult::Classification::score);

    nb::class_<ClassificationResult, ResultBase>(m, "ClassificationResult")
        .def(nb::init<>())
        .def_ro("topLabels", &ClassificationResult::topLabels)
        .def("__repr__", &ClassificationResult::operator std::string);

    nb::class_<ModelBase>(m, "ModelBase")
        .def("load", [](ModelBase& self, const std::string& device, size_t num_infer_requests) {
            auto core = ov::Core();
            self.load(core, device, num_infer_requests);
        });

    nb::class_<ImageModel, ModelBase>(m, "ImageModel");

    nb::class_<ClassificationModel, ImageModel>(m, "ClassificationModel")
        .def_static(
            "create_model",
            [](const std::string& model_path,
               const std::map<std::string, nb::object>& configuration,
               bool preload,
               const std::string& device) {
                auto ov_any_config = ov::AnyMap();
                for (const auto& item : configuration) {
                    ov_any_config[item.first] = py_object_to_any(item.second, item.first);
                }

                return ClassificationModel::create_model(model_path, ov_any_config, preload, device);
            },
            nb::arg("model_path"),
            nb::arg("configuration") = ov::AnyMap({}),
            nb::arg("preload") = true,
            nb::arg("device") = "AUTO")

        .def("__call__",
             [](ClassificationModel& self, const nb::ndarray<>& input) {
                 return self.infer(wrap_np_mat(input));
             })
        .def("infer_batch", [](ClassificationModel& self, const std::vector<nb::ndarray<>> inputs) {
            std::vector<ImageInputData> input_mats;
            input_mats.reserve(inputs.size());

            for (const auto& input : inputs) {
                input_mats.push_back(wrap_np_mat(input));
            }

            return self.inferBatch(input_mats);
        });
}
