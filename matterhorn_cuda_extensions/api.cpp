#include <torch/extension.h>

#include "stdp.h"
#include "soma.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cu_stdp", &cu_stdp, "cu_stdp");
    m.def("cu_fp_lif", &cu_fp_lif, "cu_fp_lif");
    m.def("cu_bp_lif", &cu_bp_lif, "cu_bp_lif");
}