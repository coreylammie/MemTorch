#include "cuda_runtime.h"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <math.h>
#include <torch/types.h>

#include <Eigen/Core>

__device__ int transform_3d_index(int x, int y, int z, int len_y, int len_z) {
  return x * len_y * len_z + y * len_z + z;
}

__global__ void tile_matmul_kernel(
    float *mat_a_tiles_accessor,
    torch::PackedTensorAccessor32<float, 1> mat_a_tiles_map_accessor,
    const int64_t *mat_a_tiles_shape, float *mat_b_tiles_accessor,
    torch::PackedTensorAccessor32<float, 2> mat_b_tiles_map_accessor,
    const int64_t *mat_b_tiles_shape, int limit_i, int limit_j, int limit_k) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;
  if (i < limit_i && j < limit_j && k < limit_k) {
    Eigen::MatrixXf tile_a = Eigen::Map<Eigen::MatrixXf>(
        &mat_a_tiles_accessor[transform_3d_index(mat_a_tiles_map_accessor[j], i,
                                                 0, mat_a_tiles_shape[1],
                                                 mat_a_tiles_shape[2])],
        1, mat_a_tiles_shape[2]);
    Eigen::MatrixXf tile_b = Eigen::Map<Eigen::MatrixXf, Eigen::RowMajor,
                                        Eigen::Stride<1, Eigen::Dynamic>>(
        &mat_b_tiles_accessor[transform_3d_index(mat_b_tiles_map_accessor[j][k],
                                                 0, 0, mat_b_tiles_shape[1],
                                                 mat_b_tiles_shape[2])],
        mat_b_tiles_shape[1], mat_b_tiles_shape[2],
        Eigen::Stride<1, Eigen::Dynamic>(1, mat_b_tiles_shape[2]));
    Eigen::VectorXf partial_sum = (tile_a * tile_b).transpose();
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
    const int64_t *mat_a_tiles_shape = mat_a_tiles.sizes().data();
    const int64_t *mat_b_tiles_shape = mat_b_tiles.sizes().data();
    float *mat_a_tiles_accessor = mat_a_tiles.data_ptr<float>();
    float *mat_b_tiles_accessor = mat_b_tiles.data_ptr<float>();
    torch::PackedTensorAccessor32<float, 1> mat_a_tiles_map_accessor =
        mat_a_tiles_map.packed_accessor32<float, 1>();
    torch::PackedTensorAccessor32<float, 2> mat_b_tiles_map_accessor =
        mat_b_tiles_map.packed_accessor32<float, 2>();
    int limit_i = mat_a_tiles.sizes().end()[-2];
    int limit_j = mat_b_tiles_map.sizes()[0];
    int limit_k = mat_b_tiles_map.sizes()[1];
    if (max_threads_dim[0] >= limit_i && max_threads_dim[1] >= limit_j &&
        max_threads_dim[2] >= limit_k) {
      // If multiple blocks are not required
      dim3 grid(limit_i, limit_j, limit_k);
      dim3 block(1, 1, 1);
      tile_matmul_kernel<<<grid, block>>>(
          mat_a_tiles_accessor, mat_a_tiles_map_accessor, mat_a_tiles_shape,
          mat_b_tiles_accessor, mat_b_tiles_map_accessor, mat_b_tiles_shape,
          limit_i, limit_j, limit_k);
    } else {
      // If multiple blocks are required
      dim3 grid(max_threads_dim[0], max_threads_dim[1], max_threads_dim[2]);
      dim3 block(ceil_int_div(limit_i, max_threads_dim[0]),
                 ceil_int_div(limit_j, max_threads_dim[1]),
                 ceil_int_div(limit_k, max_threads_dim[2]));
      tile_matmul_kernel<<<grid, block>>>(
          mat_a_tiles_accessor, mat_a_tiles_map_accessor, mat_a_tiles_shape,
          mat_b_tiles_accessor, mat_b_tiles_map_accessor, mat_b_tiles_shape,
          limit_i, limit_j, limit_k);
    }
    cudaDeviceSynchronize();
    cudaStreamSynchronize(at::cuda::getCurrentCUDAStream());
  }
  return mat_a_tiles; // TEMP
}