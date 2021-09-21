#include "cuda_runtime.h"
#include "utils.cuh"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <iostream>
#include <limits>
#include <math.h>
#include <torch/types.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/SparseQR>

#include "quantize.cuh"
#include "solve_passive.cuh"
#include "solve_sparse_linear.h"

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
    for (int ii = 0; ii < partial_sum.size(); ii++) {
      result[transform_2d_index(i, j * mat_b_tiles_shape[2] + ii,
                                mat_b_shape_back)] += partial_sum[ii];
    }
    free(&partial_sum);
  }
}

__global__ void tile_matmul_kernel_A(
    float *mat_a_tiles_accessor,
    torch::PackedTensorAccessor32<float, 1> mat_a_tiles_map_accessor,
    int64_t *mat_a_tiles_shape, float *mat_b_tiles_accessor,
    torch::PackedTensorAccessor32<float, 2> mat_b_tiles_map_accessor,
    int64_t *mat_b_tiles_shape, int mat_b_shape_back,
    int *ABCD_matrix_indices_x, int *ABCD_matrix_indices_y,
    double *ABCD_matrix_values, int *ABCD_matrix_compressed_rows,
    int *ABCD_matrix_compressed_columns, double *ABCD_matrix_compressed_values,
    double *E_matrix, float source_resistance, float line_resistance,
    int limit_i, int limit_j, int limit_k) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;
  if (i < limit_i && j < limit_j && k < limit_k) {
    Eigen::Map<Eigen::VectorXf> tile_a(
        &mat_a_tiles_accessor[transform_3d_index(mat_a_tiles_map_accessor[k], i,
                                                 0, mat_a_tiles_shape[1],
                                                 mat_a_tiles_shape[2])],
        mat_a_tiles_shape[1]);
    Eigen::Map<Eigen::MatrixXf, Eigen::RowMajor,
               Eigen::Stride<1, Eigen::Dynamic>>
        tile_b(&mat_b_tiles_accessor[transform_3d_index(
                   mat_b_tiles_map_accessor[k][j], 0, 0, mat_b_tiles_shape[1],
                   mat_b_tiles_shape[2])],
               mat_b_tiles_shape[1], mat_b_tiles_shape[2],
               Eigen::Stride<1, Eigen::Dynamic>(1, mat_b_tiles_shape[2]));
    int m = (int)mat_b_tiles_shape[1];
    int n = (int)mat_b_tiles_shape[2];
    int nonzero_elements = 8 * m * n - 2 * m - 2 * n;
    int kernel_index = transform_3d_index(i, j, k, limit_j, limit_k);
    construct_ABCD_E(
        tile_b, tile_a, Eigen::VectorXf::Zero(n), source_resistance,
        line_resistance,
        &ABCD_matrix_indices_x[kernel_index * nonzero_elements],
        &ABCD_matrix_indices_y[kernel_index * nonzero_elements],
        &ABCD_matrix_values[kernel_index * nonzero_elements],
        &ABCD_matrix_compressed_rows[kernel_index * nonzero_elements],
        &ABCD_matrix_compressed_columns[kernel_index * (2 * m * n)],
        &ABCD_matrix_compressed_values[kernel_index * nonzero_elements],
        &E_matrix[kernel_index * (2 * m * n)]);
  }
}

__global__ void tile_matmul_kernel_B(
    double *E_matrix, float *mat_b_tiles_accessor,
    torch::PackedTensorAccessor32<float, 2> mat_b_tiles_map_accessor,
    int64_t *mat_b_tiles_shape, int mat_b_shape_back, int m, int n, int limit_i,
    int limit_j, int limit_k, float *result) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;
  if (i < limit_i && j < limit_j && k < limit_k) {
    int kernel_index = transform_3d_index(i, j, k, limit_j, limit_k);
    Eigen::Map<Eigen::MatrixXf, Eigen::RowMajor,
               Eigen::Stride<1, Eigen::Dynamic>>
        tile_b(&mat_b_tiles_accessor[transform_3d_index(
                   mat_b_tiles_map_accessor[k][j], 0, 0, mat_b_tiles_shape[1],
                   mat_b_tiles_shape[2])],
               mat_b_tiles_shape[1], mat_b_tiles_shape[2],
               Eigen::Stride<1, Eigen::Dynamic>(1, mat_b_tiles_shape[2]));
    Eigen::MatrixXf I_applied_tensor = Eigen::MatrixXf::Zero(m, n);
    for (int ii = 0; ii < m; ii++) {
      for (int jj = 0; jj < n; jj++) {
        I_applied_tensor(ii, jj) =
            ((float)E_matrix[kernel_index * (2 * m * n) + n * ii + jj] -
             (float)
                 E_matrix[kernel_index * (2 * m * n) + m * n + n * ii + jj]) *
            tile_b(ii, jj);
      }
    }
    Eigen::VectorXf I_tensor = I_applied_tensor.colwise().sum();
    for (int ii = 0; ii < n; ii++) {
      result[transform_2d_index(i, j * mat_b_tiles_shape[2] + ii,
                                mat_b_shape_back)] += I_tensor[ii];
    }
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

__global__ void tile_matmul_kernel_B(
    double *E_matrix, float *mat_b_tiles_accessor,
    torch::PackedTensorAccessor32<float, 2> mat_b_tiles_map_accessor,
    int64_t *mat_b_tiles_shape, int mat_b_shape_back, int ADC_resolution,
    float overflow_rate, int quant_method, int m, int n, int limit_i,
    int limit_j, int limit_k, float *result) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;
  if (i < limit_i && j < limit_j && k < limit_k) {
    int kernel_index = transform_3d_index(i, j, k, limit_j, limit_k);
    Eigen::Map<Eigen::MatrixXf, Eigen::RowMajor,
               Eigen::Stride<1, Eigen::Dynamic>>
        tile_b(&mat_b_tiles_accessor[transform_3d_index(
                   mat_b_tiles_map_accessor[k][j], 0, 0, mat_b_tiles_shape[1],
                   mat_b_tiles_shape[2])],
               mat_b_tiles_shape[1], mat_b_tiles_shape[2],
               Eigen::Stride<1, Eigen::Dynamic>(1, mat_b_tiles_shape[2]));
    Eigen::MatrixXf I_applied_tensor = Eigen::MatrixXf::Zero(m, n);
    for (int ii = 0; ii < m; ii++) {
      for (int jj = 0; jj < n; jj++) {
        I_applied_tensor(ii, jj) =
            ((float)E_matrix[kernel_index * (2 * m * n) + n * ii + jj] -
             (float)
                 E_matrix[kernel_index * (2 * m * n) + m * n + n * ii + jj]) *
            tile_b(ii, jj);
      }
    }
    Eigen::VectorXf I_tensor = I_applied_tensor.colwise().sum();
    I_tensor = quantize(I_tensor, ADC_resolution, overflow_rate, quant_method);
    for (int ii = 0; ii < n; ii++) {
      result[transform_2d_index(i, j * mat_b_tiles_shape[2] + ii,
                                mat_b_shape_back)] += I_tensor[ii];
    }
  }
}

at::Tensor tile_matmul(at::Tensor mat_a_tiles, at::Tensor mat_a_tiles_map,
                       int mat_a_shape[2], at::Tensor mat_b_tiles,
                       at::Tensor mat_b_tiles_map, int mat_b_shape[2],
                       int ADC_resolution, float overflow_rate,
                       int quant_method, float source_resistance,
                       float line_resistance, int cuda_malloc_heap_size) {
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
                     size_t(1024) * size_t(1024) *
                         size_t(cuda_malloc_heap_size));
  dim3 grid;
  dim3 block;
  if (max_threads_dim[0] >= limit_i && max_threads_dim[1] >= limit_j &&
      max_threads_dim[2] >= limit_k) {
    // If multiple blocks are not required
    grid = {(unsigned int)limit_i, (unsigned int)limit_j,
            (unsigned int)limit_k};
    block = {1, 1, 1};
  } else {
    // If multiple blocks are required
    grid = {(unsigned int)max_threads_dim[0], (unsigned int)max_threads_dim[1],
            (unsigned int)max_threads_dim[2]};
    block = {(unsigned int)ceil_int_div(limit_i, max_threads_dim[0]),
             (unsigned int)ceil_int_div(limit_j, max_threads_dim[1]),
             (unsigned int)ceil_int_div(limit_k, max_threads_dim[2])};
  }
  if (line_resistance == -1) {
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
    int m = mat_b_tiles_shape_host[1];
    int n = mat_b_tiles_shape_host[2];
    int non_zero_elements = 8 * m * n - 2 * m - 2 * n;
    int n_kernels = grid.x * block.x * grid.y * block.y * grid.z * block.z;
    int *ABCD_matrix_indices_x;
    int *ABCD_matrix_indices_y;
    double *ABCD_matrix_values;
    int *ABCD_matrix_compressed_columns;
    int *ABCD_matrix_compressed_rows;
    double *ABCD_matrix_compressed_values;
    double *E_matrix;
    cudaSafeCall(cudaMalloc(&ABCD_matrix_indices_x,
                            sizeof(int) * non_zero_elements * n_kernels));
    cudaSafeCall(cudaMalloc(&ABCD_matrix_indices_y,
                            sizeof(int) * non_zero_elements * n_kernels));
    cudaSafeCall(cudaMalloc(&ABCD_matrix_values,
                            sizeof(double) * non_zero_elements * n_kernels));
    cudaSafeCall(cudaMalloc(&ABCD_matrix_compressed_columns,
                            sizeof(int) * (2 * n * m) * n_kernels));
    cudaSafeCall(cudaMalloc(&ABCD_matrix_compressed_rows,
                            sizeof(int) * non_zero_elements * n_kernels));
    cudaSafeCall(cudaMalloc(&ABCD_matrix_compressed_values,
                            sizeof(double) * non_zero_elements * n_kernels));
    cudaSafeCall(
        cudaMalloc(&E_matrix, sizeof(double) * (2 * m * n) * n_kernels));
    tile_matmul_kernel_A<<<grid, block>>>(
        mat_a_tiles_accessor, mat_a_tiles_map_accessor, mat_a_tiles_shape,
        mat_b_tiles_accessor, mat_b_tiles_map_accessor, mat_b_tiles_shape,
        mat_b_shape[1], ABCD_matrix_indices_x, ABCD_matrix_indices_y,
        ABCD_matrix_values, ABCD_matrix_compressed_rows,
        ABCD_matrix_compressed_columns, ABCD_matrix_compressed_values, E_matrix,
        source_resistance, line_resistance, limit_i, limit_j, limit_k);
    cudaSafeCall(cudaDeviceSynchronize());
    cudaSafeCall(cudaFree(ABCD_matrix_indices_x));
    cudaSafeCall(cudaFree(ABCD_matrix_indices_y));
    cudaSafeCall(cudaFree(ABCD_matrix_values));
    int *ABCD_matrix_compressed_rows_host =
        (int *)malloc(sizeof(int) * non_zero_elements);
    int *ABCD_matrix_compressed_columns_host =
        (int *)malloc(sizeof(int) * (2 * m * n));
    double *ABCD_matirx_compressed_values_host =
        (double *)malloc(sizeof(double) * non_zero_elements);
    double *E_matrix_host =
        (double *)malloc(sizeof(double) * (2 * m * n) * n_kernels);
    cudaSafeCall(cudaMemcpy(E_matrix_host, E_matrix,
                            sizeof(double) * (2 * m * n) * n_kernels,
                            cudaMemcpyDeviceToHost));
#pragma omp parallel for
    for (int i = 0; i < n_kernels; i++) {
      cudaSafeCall(
          cudaMemcpy(ABCD_matrix_compressed_rows_host,
                     &ABCD_matrix_compressed_rows[i * non_zero_elements],
                     sizeof(int) * non_zero_elements, cudaMemcpyDeviceToHost));
      cudaSafeCall(cudaMemcpy(ABCD_matrix_compressed_columns_host,
                              &ABCD_matrix_compressed_columns[i * (2 * n * m)],
                              sizeof(int) * (2 * m * n),
                              cudaMemcpyDeviceToHost));
      cudaSafeCall(cudaMemcpy(
          ABCD_matirx_compressed_values_host,
          &ABCD_matrix_compressed_values[i * non_zero_elements],
          sizeof(double) * non_zero_elements, cudaMemcpyDeviceToHost));
      Eigen::Map<Eigen::SparseMatrix<double>> A(
          (2 * m * n), (2 * m * n), non_zero_elements,
          ABCD_matrix_compressed_columns_host, ABCD_matrix_compressed_rows_host,
          ABCD_matirx_compressed_values_host);
      solve_sparse_linear(A, &E_matrix_host[i * (2 * n * m)], 2 * m * n);
    }
    free(ABCD_matrix_compressed_rows_host);
    free(ABCD_matrix_compressed_columns_host);
    free(ABCD_matirx_compressed_values_host);
    cudaSafeCall(cudaMemcpy(E_matrix, E_matrix_host,
                            sizeof(double) * (2 * n * m) * n_kernels,
                            cudaMemcpyHostToDevice));
    free(E_matrix_host);
    if (ADC_resolution == -1) {
      tile_matmul_kernel_B<<<grid, block>>>(
          E_matrix, mat_b_tiles_accessor, mat_b_tiles_map_accessor,
          mat_b_tiles_shape, mat_b_shape[1], m, n, limit_i, limit_j, limit_k,
          result.data_ptr<float>());
    } else {
      tile_matmul_kernel_B<<<grid, block>>>(
          E_matrix, mat_b_tiles_accessor, mat_b_tiles_map_accessor,
          mat_b_tiles_shape, mat_b_shape[1], ADC_resolution, overflow_rate,
          quant_method, m, n, limit_i, limit_j, limit_k,
          result.data_ptr<float>());
    }
    cudaSafeCall(cudaFree(E_matrix));
  }
  cudaSafeCall(cudaDeviceSynchronize());
  cudaSafeCall(cudaFree(mat_a_tiles_shape));
  cudaSafeCall(cudaFree(mat_b_tiles_shape));
  cudaStreamSynchronize(at::cuda::getCurrentCUDAStream());
  return result;
}