#include <ATen/ATen.h>
#include <cmath>
#include <torch/extension.h>

#include "gen_tiles.h"
#include "inference.h"
#include "quantize.h"
#include "solve_passive.h"
#include "tile_matmul.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  quantize_bindings(m);
  gen_tiles_bindings(m);
  tile_matmul_bindings(m);
  inference_bindings(m);
  solve_passive_bindings(m);
}