#include "cuda_runtime.h"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <iostream>
#include <limits>
#include <math.h>
#include <torch/types.h>

#include <Eigen/Core>

#include "utils.cuh"

#include "quantize.cuh"

__global__ void tile_matmul_kernel(
    float *mat_a_tiles_accessor,
    torch::PackedTensorAccessor32<float, 1> mat_a_tiles_map_accessor,
    int64_t *mat_a_tiles_shape, float *mat_b_tiles_accessor,
    torch::PackedTensorAccessor32<float, 2> mat_b_tiles_map_accessor,
    int64_t *mat_b_tiles_shape, int mat_b_shape_back, int limit_i, int limit_j,
    int limit_k, float *result) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;
  if (i < limit_i && j < limit_j && k < limit_k) {
    Eigen::Map<Eigen::MatrixXf> tile_a(
        &mat_a_tiles_accessor[transform_3d_index(mat_a_tiles_map_accessor[k], i,
                                                 0, mat_a_tiles_shape[1],
                                                 mat_a_tiles_shape[2])],
        1, mat_a_tiles_shape[2]);
    Eigen::Map<Eigen::MatrixXf, Eigen::RowMajor,
               Eigen::Stride<1, Eigen::Dynamic>>
        tile_b(&mat_b_tiles_accessor[transform_3d_index(
                   mat_b_tiles_map_accessor[k][j], 0, 0, mat_b_tiles_shape[1],
                   mat_b_tiles_shape[2])],
               mat_b_tiles_shape[1], mat_b_tiles_shape[2],
               Eigen::Stride<1, Eigen::Dynamic>(1, mat_b_tiles_shape[2]));
    Eigen::VectorXf partial_sum = (tile_a * tile_b).transpose();
#pragma omp parallel for
    for (int ii = 0; ii < partial_sum.size(); ii++) {
      result[transform_2d_index(i, j * mat_b_tiles_shape[2] + ii,
                                mat_b_shape_back)] += partial_sum[ii];
    }
    free(&partial_sum);
  }
}

__global__ void tile_matmul_kernel(
    float *mat_a_tiles_accessor,
    torch::PackedTensorAccessor32<float, 1> mat_a_tiles_map_accessor,
    int64_t *mat_a_tiles_shape, float *mat_b_tiles_accessor,
    torch::PackedTensorAccessor32<float, 2> mat_b_tiles_map_accessor,
    int64_t *mat_b_tiles_shape, int mat_b_shape_back, int ADC_resolution,
    float overflow_rate, int quant_method, int limit_i, int limit_j,
    int limit_k, float *result) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;
  if (i < limit_i && j < limit_j && k < limit_k) {
    Eigen::Map<Eigen::MatrixXf> tile_a(
        &mat_a_tiles_accessor[transform_3d_index(mat_a_tiles_map_accessor[k], i,
                                                 0, mat_a_tiles_shape[1],
                                                 mat_a_tiles_shape[2])],
        1, mat_a_tiles_shape[2]);

    Eigen::Map<Eigen::MatrixXf, Eigen::RowMajor,
               Eigen::Stride<1, Eigen::Dynamic>>
        tile_b(&mat_b_tiles_accessor[transform_3d_index(
                   mat_b_tiles_map_accessor[k][j], 0, 0, mat_b_tiles_shape[1],
                   mat_b_tiles_shape[2])],
               mat_b_tiles_shape[1], mat_b_tiles_shape[2],
               Eigen::Stride<1, Eigen::Dynamic>(1, mat_b_tiles_shape[2]));
    Eigen::VectorXf partial_sum = (tile_a * tile_b).transpose();
    partial_sum =
        quantize(partial_sum, ADC_resolution, overflow_rate, quant_method);
#pragma omp parallel for
    for (int ii = 0; ii < partial_sum.size(); ii++) {
      result[transform_2d_index(i, j * mat_b_tiles_shape[2] + ii,
                                mat_b_shape_back)] += partial_sum[ii];
    }
    free(&partial_sum);
  }
}

at::Tensor tile_matmul(at::Tensor mat_a_tiles, at::Tensor mat_a_tiles_map,
                       int mat_a_shape[2], at::Tensor mat_b_tiles,
                       at::Tensor mat_b_tiles_map, int mat_b_shape[2],
                       int ADC_resolution, float overflow_rate,
                       int quant_method, int cuda_malloc_heap_size) {
  assert(at::cuda::is_available());
  mat_a_tiles = mat_a_tiles.to(torch::Device("cuda:0"));
  mat_a_tiles_map = mat_a_tiles_map.to(torch::Device("cuda:0"));
  mat_b_tiles = mat_b_tiles.to(torch::Device("cuda:0"));
  mat_b_tiles_map = mat_b_tiles_map.to(torch::Device("cuda:0"));
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int *max_threads_dim = prop.maxThreadsDim;
  int64_t *mat_a_tiles_shape_host = (int64_t *)malloc(sizeof(int64_t) * 3);
  int64_t *mat_b_tiles_shape_host = (int64_t *)malloc(sizeof(int64_t) * 3);
  for (int i = 0; i < 3; i++) {
    mat_a_tiles_shape_host[i] = mat_a_tiles.sizes()[i];
    mat_b_tiles_shape_host[i] = mat_b_tiles.sizes()[i];
  }
  int64_t *mat_a_tiles_shape;
  int64_t *mat_b_tiles_shape;
  cudaSafeCall(cudaMalloc(&mat_a_tiles_shape, sizeof(int64_t) * 3));
  cudaSafeCall(cudaMalloc(&mat_b_tiles_shape, sizeof(int64_t) * 3));
  cudaSafeCall(cudaMemcpy(mat_a_tiles_shape, mat_a_tiles_shape_host,
                          sizeof(int64_t) * 3, cudaMemcpyHostToDevice));
  cudaSafeCall(cudaMemcpy(mat_b_tiles_shape, mat_b_tiles_shape_host,
                          sizeof(int64_t) * 3, cudaMemcpyHostToDevice));
  float *mat_a_tiles_accessor = mat_a_tiles.data_ptr<float>();
  float *mat_b_tiles_accessor = mat_b_tiles.data_ptr<float>();
  torch::PackedTensorAccessor32<float, 1> mat_a_tiles_map_accessor =
      mat_a_tiles_map.packed_accessor32<float, 1>();
  torch::PackedTensorAccessor32<float, 2> mat_b_tiles_map_accessor =
      mat_b_tiles_map.packed_accessor32<float, 2>();
  int limit_i = mat_a_tiles.sizes().end()[-2];
  int limit_j = mat_b_tiles_map.sizes()[1];
  int limit_k = mat_b_tiles_map.sizes()[0];
  at::Tensor result =
      at::zeros({mat_a_shape[0], mat_b_shape[1]}, torch::device(torch::kCUDA));
  cudaDeviceSetLimit(cudaLimitMallocHeapSize,
                     1024 * 1024 * cuda_malloc_heap_size);
  if (max_threads_dim[0] >= limit_i && max_threads_dim[1] >= limit_j &&
      max_threads_dim[2] >= limit_k) {
    // If multiple blocks are not required
    dim3 grid(limit_i, limit_j, limit_k);
    dim3 block(1, 1, 1);
    if (ADC_resolution == -1) {
      tile_matmul_kernel<<<grid, block>>>(
          mat_a_tiles_accessor, mat_a_tiles_map_accessor, mat_a_tiles_shape,
          mat_b_tiles_accessor, mat_b_tiles_map_accessor, mat_b_tiles_shape,
          mat_b_shape[1], limit_i, limit_j, limit_k, result.data_ptr<float>());
    } else {
      tile_matmul_kernel<<<grid, block>>>(
          mat_a_tiles_accessor, mat_a_tiles_map_accessor, mat_a_tiles_shape,
          mat_b_tiles_accessor, mat_b_tiles_map_accessor, mat_b_tiles_shape,
          mat_b_shape[1], ADC_resolution, overflow_rate, quant_method, limit_i,
          limit_j, limit_k, result.data_ptr<float>());
    }
  } else {
    // If multiple blocks are required
    dim3 grid(max_threads_dim[0], max_threads_dim[1], max_threads_dim[2]);
    dim3 block(ceil_int_div(limit_i, max_threads_dim[0]),
               ceil_int_div(limit_j, max_threads_dim[1]),
               ceil_int_div(limit_k, max_threads_dim[2]));
    if (ADC_resolution == -1) {
      tile_matmul_kernel<<<grid, block>>>(
          mat_a_tiles_accessor, mat_a_tiles_map_accessor, mat_a_tiles_shape,
          mat_b_tiles_accessor, mat_b_tiles_map_accessor, mat_b_tiles_shape,
          mat_b_shape[1], limit_i, limit_j, limit_k, result.data_ptr<float>());
    } else {
      tile_matmul_kernel<<<grid, block>>>(
          mat_a_tiles_accessor, mat_a_tiles_map_accessor, mat_a_tiles_shape,
          mat_b_tiles_accessor, mat_b_tiles_map_accessor, mat_b_tiles_shape,
          mat_b_shape[1], ADC_resolution, overflow_rate, quant_method, limit_i,
          limit_j, limit_k, result.data_ptr<float>());
    }
  }
  cudaSafeCall(cudaDeviceSynchronize());
  cudaSafeCall(cudaFree(mat_a_tiles_shape));
  cudaSafeCall(cudaFree(mat_b_tiles_shape));
  cudaStreamSynchronize(at::cuda::getCurrentCUDAStream());
  return result;
}