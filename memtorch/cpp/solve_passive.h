#include <ATen/ATen.h>
#include <cmath>
#include <torch/extension.h>

void solve_passive_bindings(py::module_ &m);
at::Tensor solve_passive(at::Tensor conductance_matrix, at::Tensor V_WL,
                         at::Tensor V_BL, float R_source, float R_line,
                         bool det_readout_currents);
at::Tensor solve_passive(at::Tensor conductance_matrix, at::Tensor V_WL,
                         at::Tensor V_BL, float R_source, float R_line,
                         int n_input_batches);