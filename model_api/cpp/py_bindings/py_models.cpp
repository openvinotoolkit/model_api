#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "models/results.h"
#include "models/classification_model.h"

namespace nb = nanobind;

NB_MODULE(py_model_api, m) {
    m.doc() = "Nanobind binding for OpenVINO Vision API library";
    nb::class_<ResultBase>(m, "ResultBase")
        .def(nb::init<>());

    nb::class_<ClassificationResult::Classification>(m, "Classification")
        .def(nb::init<unsigned int, const std::string, float>())
        .def_rw("id", &ClassificationResult::Classification::id)
        .def_rw("label", &ClassificationResult::Classification::label)
        .def_rw("score", &ClassificationResult::Classification::score);

    nb::class_<ClassificationResult, ResultBase>(m, "ClassificationResult")
        .def(nb::init<>())
        .def_ro("topLabels", &ClassificationResult::topLabels)
        .def("__repr__", &ClassificationResult::operator std::string);


    nb::class_<ClassificationModel>(m, "ClassificationModel")
        .def("create_model", nb::overload_cast<const std::string&, const ov::AnyMap&, bool, const std::string&>(&ClassificationModel::create_model));
}
