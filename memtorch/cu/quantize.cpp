#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <iostream>
#include <torch/extension.h>
#include <vector>

#include "quantize_kernels.cuh"

void quantize_bindings(py::module_ &m) {
  m.def("quantize", [&](at::Tensor tensor, int bits, float overflow_rate,
                        int quant_method, float min, float max) {
    return quantize(tensor, bits, overflow_rate, quant_method, min, max);
  });
  m.def(
      "quantize",
      [&](at::Tensor tensor, int bits, float overflow_rate, int quant_method,
          float min, float max) {
        return quantize(tensor, bits, overflow_rate, quant_method, min, max);
      },
      py::arg("tensor"), py::arg("bits"), py::arg("overflow_rate"),
      py::arg("quant_method"), py::arg("min") = NULL, py::arg("max") = NULL);
}