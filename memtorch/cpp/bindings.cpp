#include <ATen/ATen.h>
#include <cmath>
#include <torch/extension.h>

#include "gen_tiles.h"
#include "inference.h"
#include "quantize.h"
#include "readout_passive.h"
#include "solve_sparse_linear.h"
#include "tile_matmul.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  quantize_bindings(m);
  gen_tiles_bindings(m);
  tile_matmul_bindings(m);
  inference_bindings(m);
  solve_sparse_linear_bindings(m);
  readout_passive_bindings(m);
}