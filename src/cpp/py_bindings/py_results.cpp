/*
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <openvino/openvino.hpp>

#include "models/results.h"

namespace nb = nanobind;

void init_results_modules(nb::module_& m) {

    nb::class_<DetectedObject>(m, "DetectedObject")
        .def(nb::init<>())
        .def_rw("labelID", &DetectedObject::labelID)
        .def_rw("label", &DetectedObject::label)
        .def_rw("confidence", &DetectedObject::confidence);

    nb::class_<SegmentedObject, DetectedObject>(m, "SegmentedObject")
        .def(nb::init<>())
        .def_prop_ro(
            "mask",
            [](SegmentedObject& s) {
                return nb::ndarray<uint8_t, nb::numpy, nb::c_contig>(s.mask.data,
                    {
                        static_cast<size_t>(s.mask.rows),
                        static_cast<size_t>(s.mask.cols),
                        static_cast<size_t>(s.mask.channels())
                    });
            },
            nb::rv_policy::reference_internal
        );
}
