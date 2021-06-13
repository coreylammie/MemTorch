#include <ATen/ATen.h>
#include <cmath>
#include <torch/extension.h>

#include "quantize.h"
#include "tile_matmul.h"

void quantize_bindings(py::module_ &);
void tile_matmul_bindings(py::module_ &);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  quantize_bindings(m);
  tile_matmul_bindings(m);
}