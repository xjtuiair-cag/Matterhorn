#ifndef _MATTERHORN_API
#define _MATTERHORN_API


#include <torch/extension.h>
#include "soma.h"
#include "stdp.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("stdp", &stdp, "stdp");
    m.def("fp_lif", &fp_lif, "fp_lif");
    m.def("bp_lif", &bp_lif, "bp_lif");
}


#endif