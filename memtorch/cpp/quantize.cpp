#include <ATen/ATen.h>
#include <cmath>
#include <torch/extension.h>

#include "quantize.h"

void quantize_bindings(py::module_ &m) {
  // Binding for void quantize(at::Tensor tensor, int n_quant_levels, T min, T
  // max)
  m.def(
      "quantize",
      [](at::Tensor tensor, int n_quant_levels, float min, float max) {
        if (tensor.dtype() == torch::kFloat32) {
          quantize<float>(tensor, n_quant_levels, (float)min, (float)max);
        } else {
          quantize<double>(tensor, n_quant_levels, (double)min, (double)max);
        }
        return tensor;
      },
      py::arg("tensor"), py::arg("n_quant_levels"), py::arg("min") = NULL,
      py::arg("max") = NULL);
  // Binding for for void quantize(at::Tensor tensor, int bits, T
  // overflow_rate, int quant_method, T min, T max)
  m.def(
      "quantize",
      [](at::Tensor tensor, int bits, float overflow_rate, int quant_method,
         float min, float max) {
        if (tensor.dtype() == torch::kFloat32) {
          return quantize<float>(tensor, bits, (float)overflow_rate,
                                 quant_method, (float)min, (float)max);
        } else {
          return quantize<double>(tensor, bits, (double)overflow_rate,
                                  quant_method, (double)min, (double)max);
        }
      },
      py::arg("tensor"), py::arg("bits"), py::arg("overflow_rate") = 0.,
      py::arg("quant_method") = 0, py::arg("min") = NULL,
      py::arg("max") = NULL);
}