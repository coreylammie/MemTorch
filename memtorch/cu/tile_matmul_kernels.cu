#include "cuda_runtime.h"
#include "gpu.cuh"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <math.h>
#include <torch/types.h>

// int mat_a_rows = mat_a_tiles.sizes().end()[-2];
// c10::IntArrayRef mat_b_tiles_shape = mat_b_tiles.sizes();
// at::Tensor partial_sum =
//     at::zeros({mat_b_tiles_shape.end()[-2], mat_b_tiles_shape.back()});
// at::Tensor result = at::zeros({mat_a_shape[0], mat_b_shape[1]});
// c10::IntArrayRef mat_b_tiles_map_shape = mat_b_tiles_map.sizes();
// #pragma omp parallel for
// for (int i = 0; i < mat_a_rows; i++) {
//   at::Tensor mat_a_row_tiles = mat_a_tiles.index({Slice(), i, Slice()});
//   for (int j = 0; j < mat_b_tiles_map_shape[0]; j++) {
//     at::Tensor tile_a = mat_a_row_tiles[mat_a_tiles_map[j].item<int>()];
//     for (int k = 0; k < mat_b_tiles_map_shape[1]; k++) {
//       at::Tensor tile_b = mat_b_tiles[mat_b_tiles_map[j][k].item<int>()];
//       partial_sum[k] += at::matmul(tile_a, tile_b).squeeze();
//     }
//   }
//   result.index_put_({i, Slice()},
//                     partial_sum.flatten().index({Slice(0, mat_b_shape[1])}));
//   partial_sum = partial_sum.zero_();
// }
// return result;
// }

// __device__ at::Tensor tile_matmul_inner_inner_kernel() {
//   int k = threadIdx.z + blockIdx.z * blockDim.z;
//   // Inner loop logic
// }

__global__ void
tile_matmul_kernel(torch::PackedTensorAccessor32<float, 3> tensor, int limit_y,
                   int limit_z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;
  tensor[i][j][k] = 1.0f;
}

at::Tensor tile_matmul(at::Tensor mat_a_tiles, at::Tensor mat_a_tiles_map,
                       int mat_a_shape[2], at::Tensor mat_b_tiles,
                       at::Tensor mat_b_tiles_map, int mat_b_shape[2]) {
  if (at::cuda::is_available()) {
    mat_a_tiles.to(torch::Device("cuda:0"));
    mat_a_tiles_map.to(torch::Device("cuda:0"));
    mat_b_tiles.to(torch::Device("cuda:0"));
    // mat_b_tiles_map.to(torch::Device("cuda:0"));
    // int mat_a_rows = mat_a_tiles.sizes().end()[-2];
    c10::IntArrayRef mat_b_tiles_shape = mat_b_tiles.sizes();
    std::cout << mat_b_tiles_shape << std::endl;
    // at::Tensor partial_sum =
    //     at::zeros({mat_b_tiles_shape.end()[-2], mat_b_tiles_shape.back()});
    // at::Tensor result = at::zeros({mat_a_shape[0], mat_b_shape[1]});
    // c10::IntArrayRef mat_b_tiles_map_shape = mat_b_tiles_map.sizes();
    std::cout << "here." << std::endl;
    mat_b_tiles = at::zeros({32, 8, 8}).to(torch::Device("cuda:0"));
    // int thread_limit = 1024;
    // int n_threads = 32 * 16 * 16;
    // int n_blocks = (n_threads / thread_limit) + 1;
    dim3 grid(32, 8, 8);
    dim3 block(1, 1, 1);

    tile_matmul_kernel<<<grid, block>>>(
        mat_b_tiles.packed_accessor32<float, 3>(), 8, 8);
    cudaDeviceSynchronize();
    cudaStreamSynchronize(at::cuda::getCurrentCUDAStream());
    std::cout << "done." << std::endl;
    std::cout << mat_b_tiles << std::endl;
  }
  return mat_a_tiles;
}