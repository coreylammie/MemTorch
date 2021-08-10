#include <ATen/ATen.h>
#include <cmath>
#include <torch/extension.h>

#include "gen_tiles.h"
#include "inference.h"
#include "quantize.h"
#include "tile_matmul.h"

void quantize_bindings(py::module_ &);
void gen_tiles_bindings(py::module_ &);
void tile_matmul_bindings(py::module_ &);
void inference_bindings(py::module_ &);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  quantize_bindings(m);
  gen_tiles_bindings(m);
  tile_matmul_bindings(m);
  inference_bindings(m);
}