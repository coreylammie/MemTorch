#include "cuda_runtime.h"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <math.h>
#include <torch/types.h>

#include <Eigen/Core>

__global__ void
tile_matmul_kernel(torch::PackedTensorAccessor32<float, 3> tensor, int limit_x,
                   int limit_y, int limit_z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;
  if (i < limit_x && j < limit_y && k < limit_z) {
    // auto mat_a_row_tiles = tensor[][i][];
    Eigen::Matrix<float, 3, 3> t1;
    t1.setZero();
    t1(1, 1) = 5.0f;
    printf("%f", t1(0, 0));
    printf("%f\n", t1(1, 1));
    tensor[i][j][k] = 1.0f;
  }
}

int ceil_int_div(int a, int b) { return (a + b - 1) / b; }

at::Tensor tile_matmul(at::Tensor mat_a_tiles, at::Tensor mat_a_tiles_map,
                       int mat_a_shape[2], at::Tensor mat_b_tiles,
                       at::Tensor mat_b_tiles_map, int mat_b_shape[2]) {
  if (at::cuda::is_available()) {
    mat_a_tiles.to(torch::Device("cuda:0"));
    mat_a_tiles_map.to(torch::Device("cuda:0"));
    mat_b_tiles.to(torch::Device("cuda:0"));
    mat_b_tiles_map.to(torch::Device("cuda:0"));
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("%s\n", prop.name);
    int *max_threads_dim = prop.maxThreadsDim;
    // Debugging- TEMP
    int dim_a = 1024;
    int dim_b = 1024;
    int dim_c = 62;
    mat_b_tiles = at::zeros({dim_a, dim_b, dim_c}).to(torch::Device("cuda:0"));
    if (max_threads_dim[0] >= dim_a && max_threads_dim[1] >= dim_b &&
        max_threads_dim[2] >= dim_c) {
      // If multiple blocks are not required
      dim3 grid(dim_a, dim_b, dim_c);
      dim3 block(1, 1, 1);
      printf("Grid : {%d, %d, %d} blocks. Blocks : {%d, %d, %d} threads.\n",
             grid.x, grid.y, grid.z, block.x, block.y, block.z);
      tile_matmul_kernel<<<grid, block>>>(
          mat_b_tiles.packed_accessor32<float, 3>(), dim_a, dim_b, dim_c);
    } else {
      // If multiple blocks are required
      dim3 grid(max_threads_dim[0], max_threads_dim[1], max_threads_dim[2]);
      dim3 block(ceil_int_div(dim_a, max_threads_dim[0]),
                 ceil_int_div(dim_b, max_threads_dim[1]),
                 ceil_int_div(dim_c, max_threads_dim[2]));
      printf("Grid : {%d, %d, %d} blocks. Blocks : {%d, %d, %d} threads.\n",
             grid.x, grid.y, grid.z, block.x, block.y, block.z);
      tile_matmul_kernel<<<grid, block>>>(
          mat_b_tiles.packed_accessor32<float, 3>(), dim_a, dim_b, dim_c);
    }
    cudaDeviceSynchronize();
    cudaStreamSynchronize(at::cuda::getCurrentCUDAStream());
    std::cout << mat_b_tiles.amin().item<float>()
              << std::endl; // To validate logic
  }
  return mat_a_tiles;
}