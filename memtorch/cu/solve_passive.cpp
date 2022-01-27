#include <ATen/ATen.h>
#include <cmath>
#include <torch/extension.h>

#include <Eigen/Core>

#include <Eigen/SparseCore>

#include <Eigen/SparseLU>

#include "solve_passive_kernels.cuh"

void solve_passive_bindings(py::module_ &m) {
  m.def(
      "solve_passive",
      [&](at::Tensor conductance_matrix, at::Tensor V_WL, at::Tensor V_BL,
          int ADC_resolution, float overflow_rate, int quant_method,
          float R_source, float R_line, bool det_readout_currents) {
        return solve_passive(conductance_matrix, V_WL, V_BL, ADC_resolution,
                             overflow_rate, quant_method, R_source, R_line,
                             det_readout_currents);
      },
      py::arg("conductance_matrix"), py::arg("V_WL"), py::arg("V_BL"),
      py::arg("ADC_resolution") = -1, py::arg("overflow_rate") = -1,
      py::arg("quant_method") = -1, py::arg("R_source"), 
      py::arg("R_line"), py::arg("det_readout_currents") = true);

  m.def(
      "solve_passive",
      [&](at::Tensor conductance_matrix, at::Tensor V_WL, at::Tensor V_BL,
          int ADC_resolution, float overflow_rate, int quant_method,
          float R_source, float R_line, bool det_readout_currents) {
        return solve_passive(conductance_matrix, V_WL, V_BL, ADC_resolution,
                             overflow_rate, quant_method, R_source, R_line,
                             det_readout_currents);
      },
      py::arg("conductance_matrix"), py::arg("V_WL"), py::arg("V_BL"),
      py::arg("ADC_resolution"), py::arg("overflow_rate"),
      py::arg("quant_method"), py::arg("R_source"), 
      py::arg("R_line"), py::arg("det_readout_currents") = true);
}