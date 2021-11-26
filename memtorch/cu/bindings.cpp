#include <ATen/ATen.h>
#include <cmath>
#include <torch/extension.h>

#include "gen_tiles.h"
#include "inference.h"
#include "solve_passive.h"
#include "tile_matmul.h"
#include "simulate_passive.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  gen_tiles_bindings_gpu(m);
  tile_matmul_bindings(m);
  inference_bindings(m);
  simulate_passive_bindings(m);
  solve_passive_bindings(m);
}