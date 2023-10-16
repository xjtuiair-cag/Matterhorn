#include <torch/extension.h>

#include "stdp.h"
#include "soma.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cu_stdp", &cu_stdp, "cu_stdp");
    m.def("cu_fp_lif_heaviside_hard", &cu_fp_lif_heaviside_hard, "cu_fp_lif_heaviside_hard");
    m.def("cu_bp_lif_rectangular_hard", &cu_bp_lif_rectangular_hard, "cu_bp_lif_rectangular_hard");
}