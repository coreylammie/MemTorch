#include <ATen/ATen.h>
#include <cmath>
#include <torch/extension.h>

#include "tile_matmul.h"

void tile_matmul_bindings(py::module_ &);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { tile_matmul_bindings(m); }