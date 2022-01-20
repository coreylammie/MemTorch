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
// #include "solve_passive.cuh"
// #include "solve_sparse_linear.h"
#include "solve_passive_kernels.cuh"

using namespace torch::indexing;

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

// __global__ void tile_matmul_kernel_A(
//     float *mat_a_tiles_accessor,
//     torch::PackedTensorAccessor32<float, 1> mat_a_tiles_map_accessor,
//     int64_t *mat_a_tiles_shape, float *mat_b_tiles_accessor,
//     torch::PackedTensorAccessor32<float, 2> mat_b_tiles_map_accessor,
//     int64_t *mat_b_tiles_shape, int mat_b_shape_back,
//     int *ABCD_matrix_indices_x, int *ABCD_matrix_indices_y,
//     double *ABCD_matrix_values, int *ABCD_matrix_compressed_rows,
//     int *ABCD_matrix_compressed_columns, double *ABCD_matrix_compressed_values,
//     double *E_matrix, float source_resistance, float line_resistance,
//     int limit_i, int limit_j, int limit_k) {
//   int i = threadIdx.x + blockIdx.x * blockDim.x;
//   int j = threadIdx.y + blockIdx.y * blockDim.y;
//   int k = threadIdx.z + blockIdx.z * blockDim.z;
//   if (i < limit_i && j < limit_j && k < limit_k) {
//     Eigen::Map<Eigen::VectorXf> tile_a(
//         &mat_a_tiles_accessor[transform_3d_index(mat_a_tiles_map_accessor[k], i,
//                                                  0, mat_a_tiles_shape[1],
//                                                  mat_a_tiles_shape[2])],
//         mat_a_tiles_shape[1]);
//     Eigen::Map<Eigen::MatrixXf, Eigen::RowMajor,
//                Eigen::Stride<1, Eigen::Dynamic>>
//         tile_b(&mat_b_tiles_accessor[transform_3d_index(
//                    mat_b_tiles_map_accessor[k][j], 0, 0, mat_b_tiles_shape[1],
//                    mat_b_tiles_shape[2])],
//                mat_b_tiles_shape[1], mat_b_tiles_shape[2],
//                Eigen::Stride<1, Eigen::Dynamic>(1, mat_b_tiles_shape[2]));
//     int m = (int)mat_b_tiles_shape[1];
//     int n = (int)mat_b_tiles_shape[2];
//     int nonzero_elements = 8 * m * n - 2 * m - 2 * n;
//     int kernel_index = transform_3d_index(i, j, k, limit_j, limit_k);
//     construct_ABCD_E(
//         tile_b, tile_a, Eigen::VectorXf::Zero(n), source_resistance,
//         line_resistance,
//         &ABCD_matrix_indices_x[kernel_index * nonzero_elements],
//         &ABCD_matrix_indices_y[kernel_index * nonzero_elements],
//         &ABCD_matrix_values[kernel_index * nonzero_elements],
//         &ABCD_matrix_compressed_rows[kernel_index * nonzero_elements],
//         &ABCD_matrix_compressed_columns[kernel_index * (2 * m * n)],
//         &ABCD_matrix_compressed_values[kernel_index * nonzero_elements],
//         &E_matrix[kernel_index * (2 * m * n)]);
//   }
// }

// __global__ void tile_matmul_kernel_B(
//     double *E_matrix, float *mat_b_tiles_accessor,
//     torch::PackedTensorAccessor32<float, 2> mat_b_tiles_map_accessor,
//     int64_t *mat_b_tiles_shape, int mat_b_shape_back, int m, int n, int limit_i,
//     int limit_j, int limit_k, float *result) {
//   int i = threadIdx.x + blockIdx.x * blockDim.x;
//   int j = threadIdx.y + blockIdx.y * blockDim.y;
//   int k = threadIdx.z + blockIdx.z * blockDim.z;
//   if (i < limit_i && j < limit_j && k < limit_k) {
//     int kernel_index = transform_3d_index(i, j, k, limit_j, limit_k);
//     Eigen::Map<Eigen::MatrixXf, Eigen::RowMajor,
//                Eigen::Stride<1, Eigen::Dynamic>>
//         tile_b(&mat_b_tiles_accessor[transform_3d_index(
//                    mat_b_tiles_map_accessor[k][j], 0, 0, mat_b_tiles_shape[1],
//                    mat_b_tiles_shape[2])],
//                mat_b_tiles_shape[1], mat_b_tiles_shape[2],
//                Eigen::Stride<1, Eigen::Dynamic>(1, mat_b_tiles_shape[2]));
//     Eigen::MatrixXf I_applied_tensor = Eigen::MatrixXf::Zero(m, n);
//     for (int ii = 0; ii < m; ii++) {
//       for (int jj = 0; jj < n; jj++) {
//         I_applied_tensor(ii, jj) =
//             ((float)E_matrix[kernel_index * (2 * m * n) + n * ii + jj] -
//              (float)
//                  E_matrix[kernel_index * (2 * m * n) + m * n + n * ii + jj]) *
//             tile_b(ii, jj);
//       }
//     }
//     Eigen::VectorXf I_tensor = I_applied_tensor.colwise().sum();
//     for (int ii = 0; ii < n; ii++) {
//       result[transform_2d_index(i, j * mat_b_tiles_shape[2] + ii,
//                                 mat_b_shape_back)] += I_tensor[ii];
//     }
//   }
// }

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


  // cudaSetDevice(0);
  // std::cout << mat_b_tiles_map << std::endl;
  // std::cout << mat_a_tiles_map << std::endl;
  // cudaDeviceSynchronize();
  // cudaThreadSynchronize();
  // std::cout << "here..." << std::endl;
  // size_t free_mem;
  // size_t total_global_mem;
  // cudaMemGetInfo(&free_mem, &total_global_mem);
  // std::cout << free_mem << "/ " << total_global_mem << std::endl;
  // free_mem *= 0.8;
  // size_t decrement = 1024 * 1024 * 1; // 1 MB
  // int *buf_d = NULL;
  // while(buf_d == NULL) {
  //   cudaFree(buf_d);
  //   cudaMalloc(&buf_d, free_mem);
  //   free_mem -= decrement;
  //   std::cout << "H" << std::endl;
  // }
  // std::cout << free_mem << std::endl;
  // std::cout << "here..." << std::endl;


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
  at::Tensor result =
      at::zeros({mat_a_shape[0], mat_b_shape[1]}, torch::device(torch::kCUDA));
// //   // cudaDeviceSetLimit(cudaLimitMallocHeapSize,
// //   //                    size_t(1024) * size_t(1024) *
// //   //                        size_t(cuda_malloc_heap_size));
  if (line_resistance == -1) {
    int limit_i = mat_a_tiles.sizes().end()[-2];
    int limit_j = mat_b_tiles_map.sizes()[1];
    int limit_k = mat_b_tiles_map.sizes()[0];
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
    // int stride = ... // TBD- stride/ limit i
    int m = mat_b_tiles_shape_host[1]; // limit j
    int n = mat_b_tiles_shape_host[2]; // limit k
    int non_zero_elements = 8 * m * n - 2 * m - 2 * n;
    int mat_a_rows = mat_a_tiles.sizes().end()[-2];
    at::Tensor partial_sum =
      at::zeros({mat_b_tiles_map.sizes()[1], mat_b_tiles_shape_host[2]}, torch::device(torch::kCUDA));
      for (int i = 0; i < mat_a_rows; i++) {
        at::Tensor mat_a_row_tiles = mat_a_tiles.index({Slice(), i, Slice()}); 
        for (int j = 0; j < mat_b_tiles_map.sizes()[0]; j++) {
          at::Tensor tile_a = mat_a_row_tiles[mat_a_tiles_map[j].item<int>()];
          for (int k = 0; k < mat_b_tiles_map.sizes()[1]; k++) {
            at::Tensor tile_b = mat_b_tiles[mat_b_tiles_map[j][k].item<int>()];
            partial_sum[k] +=
              solve_passive(tile_b, tile_a, at::zeros({tile_b.sizes()[1]}, torch::device(torch::kCUDA)),
                            source_resistance, line_resistance, true)
                  .squeeze();
          }
          result.index_put_({i, Slice()}, result.index({i, Slice()}) +
            partial_sum.flatten().index(
                {Slice(0, mat_b_shape[1])}));
          partial_sum = partial_sum.zero_();
        }
      }
      cudaSafeCall(cudaDeviceSynchronize());
  }
//   cudaSafeCall(cudaDeviceSynchronize());
//   cudaSafeCall(cudaFree(mat_a_tiles_shape));
//   cudaSafeCall(cudaFree(mat_b_tiles_shape));
//   cudaStreamSynchronize(at::cuda::getCurrentCUDAStream());
  return result;
}