#include <ATen/ATen.h>
#include <cmath>
#include <torch/extension.h>

#include "gen_ABCD_kernels.cuh"
#include "gen_tiles.h"
#include "inference.h"
#include "tile_matmul.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  gen_tiles_bindings_gpu(m);
  tile_matmul_bindings(m);
  inference_bindings(m);
  gen_ABCD_bindings(m);
}