#include "cuda_runtime.h"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <iostream>
#include <limits>
#include <math.h>
#include <torch/extension.h>
#include <torch/types.h>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <Eigen/SparseLU>

#include "solve_passive.h"
#include "solve_sparse_linear.h"
#include "utils.cuh"

class Triplet {
public:
  __host__ __device__ Triplet() : m_row(0), m_col(0), m_value(0) {}

  __host__ __device__ Triplet(int i, int j, float v)
      : m_row(i), m_col(j), m_value(v) {}

  __host__ __device__ const int &row() { return m_row; }
  __host__ __device__ const int &col() { return m_col; }
  __host__ __device__ const float &value() { return m_value; }

protected:
  int m_row, m_col;
  float m_value;
};

typedef Triplet sparse_element;

__global__ void gen_ABE_kernel(
    torch::PackedTensorAccessor32<float, 2> conductance_matrix_accessor,
    float *V_WL_accessor, float *V_BL_accessor, int m, int n, float R_source,
    float R_line, sparse_element *ABCD_matrix, float *E_matrix) {
  int i = threadIdx.x + blockIdx.x * blockDim.x; // for (int i = 0; i < m; i++)
  int j = threadIdx.y + blockIdx.y * blockDim.y; // for (int j = 0; j < n; j++)
  if (i < m && j < n) {
    int index = (i * n + j) * 5;
    // A matrix
    if (j == 0) {
      E_matrix[i * n] = V_WL_accessor[i] / R_source; // E matrix (partial)
      ABCD_matrix[index] = sparse_element(i * n, i * n,
                                          conductance_matrix_accessor[i][0] +
                                              1.0f / R_source + 1.0f / R_line);
    } else {
      ABCD_matrix[index] = sparse_element(0, 0, 0.0f);
    }
    index++;
    ABCD_matrix[index] =
        sparse_element(i * n + j, i * n + j,
                       conductance_matrix_accessor[i][j] + 2.0f / R_line);
    index++;
    if (j < n - 1) {
      ABCD_matrix[index] =
          sparse_element(i * n + j + 1, i * n + j, -1.0f / R_line);
      index++;
      ABCD_matrix[index] =
          sparse_element(i * n + j, i * n + j + 1, -1.0f / R_line);
    } else {
      ABCD_matrix[index] =
          sparse_element(i * n + j, i * n + j,
                         conductance_matrix_accessor[i][j] + 1.0 / R_line);
      index++;
      ABCD_matrix[index] = sparse_element(0, 0, 0.0f);
    }
    index++;
    // B matrix
    ABCD_matrix[index] = sparse_element(i * n + j, i * n + j + (m * n),
                                        -conductance_matrix_accessor[i][j]);
  }
}

__global__ void gen_AB_kernel(
    torch::PackedTensorAccessor32<float, 2> conductance_matrix_accessor, int m,
    int n, float R_source, float R_line, sparse_element *ABCD_matrix) {
  int i = threadIdx.x + blockIdx.x * blockDim.x; // for (int i = 0; i < m; i++)
  int j = threadIdx.y + blockIdx.y * blockDim.y; // for (int j = 0; j < n; j++)
  if (i < m && j < n) {
    int index = (i * n + j) * 5;
    // A matrix
    if (j == 0) {
      ABCD_matrix[index] = sparse_element(i * n, i * n,
                                          conductance_matrix_accessor[i][0] +
                                              1.0f / R_source + 1.0f / R_line);
    } else {
      ABCD_matrix[index] = sparse_element(0, 0, 0.0f);
    }
    index++;
    ABCD_matrix[index] =
        sparse_element(i * n + j, i * n + j,
                       conductance_matrix_accessor[i][j] + 2.0f / R_line);
    index++;
    if (j < n - 1) {
      ABCD_matrix[index] =
          sparse_element(i * n + j + 1, i * n + j, -1.0f / R_line);
      index++;
      ABCD_matrix[index] =
          sparse_element(i * n + j, i * n + j + 1, -1.0f / R_line);
    } else {
      ABCD_matrix[index] =
          sparse_element(i * n + j, i * n + j,
                         conductance_matrix_accessor[i][j] + 1.0 / R_line);
      index++;
      ABCD_matrix[index] = sparse_element(0, 0, 0.0f);
    }
    index++;
    // B matrix
    ABCD_matrix[index] = sparse_element(i * n + j, i * n + j + (m * n),
                                        -conductance_matrix_accessor[i][j]);
  }
}

__global__ void gen_CDE_kernel(
    torch::PackedTensorAccessor32<float, 2> conductance_matrix_accessor,
    float *V_WL_accessor, float *V_BL_accessor, int m, int n, float R_source,
    float R_line, sparse_element *ABCD_matrix, float *E_matrix) {
  int j = threadIdx.x + blockIdx.x * blockDim.x; // for (int j = 0; j < n; j++)
  int i = threadIdx.y + blockIdx.y * blockDim.y; // for (int i = 0; i < m; i++)
  if (j < n && i < m) {
    int index = (5 * m * n) + ((j * m + i) * 4);
    // D matrix
    if (i == 0) {
      E_matrix[m * n + (j + 1) * m - 1] =
          -V_BL_accessor[j] / R_source; // E matrix (partial)
      ABCD_matrix[index] =
          sparse_element(m * n + (j * m), m * n + j,
                         -1.0f / R_line - conductance_matrix_accessor[0][j]);
      index++;
      ABCD_matrix[index] =
          sparse_element(m * n + (j * m), m * n + j + n, 1.0f / R_line);
      index++;
      ABCD_matrix[index] = sparse_element(0, 0, 0.0f);
    } else if (i < m - 1) {
      ABCD_matrix[index] = sparse_element(
          m * n + (j * m) + i, m * n + (n * (i - 1)) + j, 1.0f / R_line);
      index++;
      ABCD_matrix[index] = sparse_element(
          m * n + (j * m) + i, m * n + (n * (i + 1)) + j, 1.0f / R_line);
      index++;
      ABCD_matrix[index] =
          sparse_element(m * n + (j * m) + i, m * n + (n * i) + j,
                         -conductance_matrix_accessor[i][j] - 2.0f / R_line);
    } else {
      ABCD_matrix[index] = sparse_element(
          m * n + (j * m) + m - 1, m * n + (n * (m - 2)) + j, 1 / R_line);
      index++;
      ABCD_matrix[index] = sparse_element(
          m * n + (j * m) + m - 1, m * n + (n * (m - 1)) + j,
          -1.0f / R_source - conductance_matrix_accessor[m - 1][j] -
              1.0f / R_line);
      index++;
      ABCD_matrix[index] = sparse_element(0, 0, 0.0f);
    }
    index++;
    // C matrix
    ABCD_matrix[index] = sparse_element(j * m + i + (m * n), n * i + j,
                                        conductance_matrix_accessor[i][j]);
  }
}

__global__ void gen_CD_kernel(
    torch::PackedTensorAccessor32<float, 2> conductance_matrix_accessor, int m,
    int n, float R_source, float R_line, sparse_element *ABCD_matrix) {
  int j = threadIdx.x + blockIdx.x * blockDim.x; // for (int j = 0; j < n; j++)
  int i = threadIdx.y + blockIdx.y * blockDim.y; // for (int i = 0; i < m; i++)
  if (j < n && i < m) {
    int index = (5 * m * n) + ((j * m + i) * 4);
    // D matrix
    if (i == 0) {
      ABCD_matrix[index] =
          sparse_element(m * n + (j * m), m * n + j,
                         -1.0f / R_line - conductance_matrix_accessor[0][j]);
      index++;
      ABCD_matrix[index] =
          sparse_element(m * n + (j * m), m * n + j + n, 1.0f / R_line);
      index++;
      ABCD_matrix[index] = sparse_element(0, 0, 0.0f);
    } else if (i < m - 1) {
      ABCD_matrix[index] = sparse_element(
          m * n + (j * m) + i, m * n + (n * (i - 1)) + j, 1.0f / R_line);
      index++;
      ABCD_matrix[index] = sparse_element(
          m * n + (j * m) + i, m * n + (n * (i + 1)) + j, 1.0f / R_line);
      index++;
      ABCD_matrix[index] =
          sparse_element(m * n + (j * m) + i, m * n + (n * i) + j,
                         -conductance_matrix_accessor[i][j] - 2.0f / R_line);
    } else {
      ABCD_matrix[index] = sparse_element(
          m * n + (j * m) + m - 1, m * n + (n * (m - 2)) + j, 1 / R_line);
      index++;
      ABCD_matrix[index] = sparse_element(
          m * n + (j * m) + m - 1, m * n + (n * (m - 1)) + j,
          -1.0f / R_source - conductance_matrix_accessor[m - 1][j] -
              1.0f / R_line);
      index++;
      ABCD_matrix[index] = sparse_element(0, 0, 0.0f);
    }
    index++;
    // C matrix
    ABCD_matrix[index] = sparse_element(j * m + i + (m * n), n * i + j,
                                        conductance_matrix_accessor[i][j]);
  }
}

__global__ void
construct_V_applied(torch::PackedTensorAccessor32<float, 2> V_applied_accessor,
                    float *V_accessor, int m, int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x; // for (int i = 0; i < m; i++)
  int j = threadIdx.y + blockIdx.y * blockDim.y; // for (int j = 0; j < n; j++)
  if (i < m && j < n) {
    V_applied_accessor[i][j] =
        V_accessor[n * i + j] - V_accessor[m * n + n * i + j];
  }
}

at::Tensor solve_passive(at::Tensor conductance_matrix, at::Tensor V_WL,
                         at::Tensor V_BL, float R_source, float R_line,
                         bool det_readout_currents) {
  assert(at::cuda::is_available());
  conductance_matrix = conductance_matrix.to(torch::Device("cuda:0"));
  V_WL = V_WL.to(torch::Device("cuda:0"));
  V_BL = V_BL.to(torch::Device("cuda:0"));
  int m = conductance_matrix.sizes()[0];
  int n = conductance_matrix.sizes()[1];
  torch::PackedTensorAccessor32<float, 2> conductance_matrix_accessor =
      conductance_matrix.packed_accessor32<float, 2>();
  float *V_WL_accessor = V_WL.data_ptr<float>();
  float *V_BL_accessor = V_BL.data_ptr<float>();
  int non_zero_elements =
      (5 * m * n) +
      (4 * m * n); // Uncompressed (with padding for CUDA execution).
  // When compressed, contains 8 * m * n - 2 * m - 2 * n unique values.
  sparse_element *ABCD_matrix;
  sparse_element *ABCD_matrix_host =
      (sparse_element *)malloc(sizeof(sparse_element) * non_zero_elements);
  cudaMalloc(&ABCD_matrix, sizeof(sparse_element) * non_zero_elements);
  float *E_matrix;
  cudaMalloc(&E_matrix, sizeof(float) * 2 * m * n);
  cudaMemset(E_matrix, 0, sizeof(float) * 2 * m * n);
  float *E_matrix_host = (float *)malloc(sizeof(float) * 2 * m * n);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int max_threads = prop.maxThreadsDim[0];
  dim3 grid;
  dim3 block;
  if (m * n > max_threads) {
    int n_grid = ceil_int_div(m * n, max_threads);
    grid = dim3(n_grid, n_grid, 1);
    block = dim3(ceil_int_div(m, n_grid), ceil_int_div(n, n_grid), 1);
  } else {
    grid = dim3(1, 1, 1);
    block = dim3(m, n, 1);
  }
  gen_ABE_kernel<<<grid, block>>>(conductance_matrix_accessor, V_WL_accessor,
                                  V_BL_accessor, m, n, R_source, R_line,
                                  ABCD_matrix, E_matrix);
  gen_CDE_kernel<<<grid, block>>>(conductance_matrix_accessor, V_WL_accessor,
                                  V_BL_accessor, m, n, R_source, R_line,
                                  ABCD_matrix, E_matrix);
  cudaSafeCall(cudaDeviceSynchronize());
  Eigen::SparseMatrix<float> ABCD(2 * m * n, 2 * m * n);
  cudaMemcpy(ABCD_matrix_host, ABCD_matrix,
             sizeof(sparse_element) * non_zero_elements,
             cudaMemcpyDeviceToHost);
  ABCD.setFromTriplets(&ABCD_matrix_host[0],
                       &ABCD_matrix_host[non_zero_elements]);
  ABCD.makeCompressed();
  cudaMemcpy(E_matrix_host, E_matrix, sizeof(float) * 2 * m * n,
             cudaMemcpyDeviceToHost);
  Eigen::Map<Eigen::VectorXf> V(E_matrix_host, 2 * m * n);
  at::Tensor V_applied_tensor =
      at::zeros({m, n}, torch::TensorOptions().device(torch::kCUDA, 0));
  torch::PackedTensorAccessor32<float, 2> V_applied_accessor =
      V_applied_tensor.packed_accessor32<float, 2>();
  float *V_accessor;
  cudaMalloc(&V_accessor, sizeof(float) * V.size());
  cudaMemcpy(V_accessor, V.data(), sizeof(float) * V.size(),
             cudaMemcpyHostToDevice);
  construct_V_applied<<<grid, block>>>(V_applied_accessor, V_accessor, m, n);
  cudaSafeCall(cudaDeviceSynchronize());
  cudaSafeCall(cudaFree(ABCD_matrix));
  cudaSafeCall(cudaFree(E_matrix));
  cudaSafeCall(cudaFree(V_accessor));
  cudaStreamSynchronize(at::cuda::getCurrentCUDAStream());
  if (!det_readout_currents) {
    return V_applied_tensor;
  } else {
    return at::sum(at::mul(V_applied_tensor, conductance_matrix), 0);
  }
}