#include <torch/extension.h>

#include "stdp.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("stdp", &stdp, "stdp");
}