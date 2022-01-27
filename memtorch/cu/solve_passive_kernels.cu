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
// #ifndef _TILE_MATMUL_KERNELS_
#include "utils.cuh"
#include "quantize.cuh"
// #endif


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
      // E matrix (partial)
      if (E_matrix != NULL) {
        if (R_source == 0) {
          E_matrix[i * n] = V_WL_accessor[i];
        } else {
          E_matrix[i * n] = V_WL_accessor[i] / R_source;
        }
      }
      if (R_source == 0) {
        ABCD_matrix[index] = sparse_element(
            i * n, i * n, conductance_matrix_accessor[i][0] + 1.0f / R_line);
      } else if (R_line == 0) {
        ABCD_matrix[index] = sparse_element(
            i * n, i * n, conductance_matrix_accessor[i][0] + 1.0f / R_source);
      } else {
        ABCD_matrix[index] =
            sparse_element(i * n, i * n,
                           conductance_matrix_accessor[i][0] + 1.0f / R_source +
                               1.0f / R_line);
      }
    } else if (j == (n - 1)) {
      if (R_line != 0) {
        sparse_element(i * n, i * n,
          conductance_matrix_accessor[i][0] + 1.0f / R_line);
      }
    } else {
      if (R_line == 0) {
        ABCD_matrix[index] = sparse_element(i * n + j, i * n + j,
                                            conductance_matrix_accessor[i][j]);
      } else {
        ABCD_matrix[index] =
            sparse_element(i * n + j, i * n + j,
                           conductance_matrix_accessor[i][j] + 2.0f / R_line);
      }
    }
    index++;
    if (j < n - 1) {
      if (R_line == 0) {
        ABCD_matrix[index] = sparse_element(i * n + j + 1, i * n + j, 0);
        index++;
        ABCD_matrix[index] = sparse_element(i * n + j, i * n + j + 1, 0);
      } else {
        ABCD_matrix[index] =
            sparse_element(i * n + j + 1, i * n + j, -1.0f / R_line);
        index++;
        ABCD_matrix[index] =
            sparse_element(i * n + j, i * n + j + 1, -1.0f / R_line);
      }
    } else {
      if (R_line == 0) {
        ABCD_matrix[index] = sparse_element(i * n + j, i * n + j,
                                            conductance_matrix_accessor[i][j]);
      } else {
        ABCD_matrix[index] =
            sparse_element(i * n + j, i * n + j,
                           conductance_matrix_accessor[i][j] + 1.0 / R_line);
      }
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
  int j = threadIdx.x + blockIdx.x * blockDim.x; // for (int j = 0; j < m; j++)
  int i = threadIdx.y + blockIdx.y * blockDim.y; // for (int i = 0; i < n; i++)
  
  if (j < m && i < n) {
    int index = (5 * m * n) + 4 * (j * n + i);
    // D matrix
    if (j == 0) {
      // E matrix (partial)
      if (E_matrix != NULL) {
        if (R_source == 0) {
          E_matrix[m * n + (j + 1) * m - 1] = -V_BL_accessor[j];
        } else {
          E_matrix[m * n + (j + 1) * m - 1] = -V_BL_accessor[j] / R_source;
        }
      }
      if (R_line == 0) {
        ABCD_matrix[index] = sparse_element(m * n + (i * m), m * n + i,
                                            -conductance_matrix_accessor[i][0]);
        index++;
        ABCD_matrix[index] = sparse_element(m * n + (i * m), m * n + i + n, 0);
      } else {
        ABCD_matrix[index] =
            sparse_element(m * n + (i * m), m * n + i,
                           -1.0f / R_line - conductance_matrix_accessor[i][0]);
        index++;
        ABCD_matrix[index] =
            sparse_element(m * n + (i * m), m * n + i + n, 1.0f / R_line);
      }
      index++;
      ABCD_matrix[index] = sparse_element(0, 0, 0.0f);
    } else if (j < (m - 1)) {
      if (R_line == 0) {
        ABCD_matrix[index] = sparse_element(
          m * n + (i * m) + j, m * n + (n * (j - 1)) + i, 0);
        index++;
        ABCD_matrix[index] = sparse_element(
          m * n + (i * m) + j, m * n + (n * j) + i, -conductance_matrix_accessor[i][j]);
        index++;
        ABCD_matrix[index] = sparse_element(
          m * n + (i * m) + j, m * n + (n * (j + 1)) + i, 0);
      } else {
        ABCD_matrix[index] = sparse_element(
          m * n + (i * m) + j, m * n + (n * (j - 1)) + i, 1.0f / R_line);
        index++;
        ABCD_matrix[index] = sparse_element(
          m * n + (i * m) + j, m * n + (n * j) + i, -conductance_matrix_accessor[i][j] - 2.0f / R_line);
        index++;
        ABCD_matrix[index] = sparse_element(
          m * n + (i * m) + j, m * n + (n * (j + 1)) + i, 1.0f / R_line);
      }
    } else {
      if (R_line != 0) {
        ABCD_matrix[index] = sparse_element(
          m * n + (i * m) + m - 1, m * n + (n * (j - 1)) + i, 1 / R_line);
      } else {
        ABCD_matrix[index] = sparse_element(m * n + (i * m) + m - 1, m * n + (n * (j - 1)) + i, 0.0f);
      }
      index++;
      if (R_source == 0) {
        ABCD_matrix[index] = sparse_element(
          m * n + (i * m) + m - 1, m * n + (n * j) + i,
            -conductance_matrix_accessor[i][m - 1] - 1.0f / R_line);
      } else if (R_line == 0) {
        ABCD_matrix[index] = sparse_element(
          m * n + (i * m) + m - 1, m * n + (n * j) + i,
            -1.0f / R_source - conductance_matrix_accessor[i][m - 1]);
      } else {
        ABCD_matrix[index] = sparse_element(
          m * n + (i * m) + m - 1, m * n + (n * j) + i,
            -1.0f / R_source - conductance_matrix_accessor[i][m - 1] -
                1.0f / R_line);
      }
      index++;
      ABCD_matrix[index] = sparse_element(0, 0, 0.0f);
    }
    index++;
    // C matrix
    ABCD_matrix[index] = sparse_element((m * n) + (i * m) + j, n * j + i,
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

__global__ void
det_sf_test(float *V_applied_tensor_accessor, int numel, int bits, float overflow_rate, float* sf) {
  float *V_applied_tensor_copy;
  V_applied_tensor_copy = (float *)malloc(numel * sizeof(float));
  for (int i = 0; i < numel; i++) {
    V_applied_tensor_copy[i] = V_applied_tensor_accessor[i];
  }
  sf[0] = det_sf(V_applied_tensor_copy, numel, bits, overflow_rate, NULL, NULL);
  delete V_applied_tensor_copy;
}

__device__ void
linear_quantize(float *V_applied_tensor_accessor, int i, float *sf, int bits, float overflow_rate) {
  float delta = powf(2.0f, sf[0]);
  float bound = powf(2.0f, bits - 1);
  float x_ = clamp_<float>(floorf((V_applied_tensor_accessor[i] / delta) + 0.5f), -bound, bound - 1) * delta;
  if (isnan(x_)) {
    V_applied_tensor_accessor[i] = 0.0f;
  } else {
    V_applied_tensor_accessor[i] = x_;
  }
}

__global__ void
quantize_test(float *V_applied_tensor_accessor, int numel, int bits, float* sf) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  bool quant_method = 1;
  if (i < numel) {
    // assume linear for now...
    if (quant_method == 0) {
      // linear
      linear_quantize(V_applied_tensor_accessor, i, sf, bits, 0.0f);
    } else if (quant_method  == 1) {
      // log
      bool s = V_applied_tensor_accessor[i] >= 0.0f;
      V_applied_tensor_accessor[i] = max_<float>(logf(abs_<float>(V_applied_tensor_accessor[i])), 1e-20f);
      linear_quantize(V_applied_tensor_accessor, i, sf, bits, 0.0f);
      if (s) {
        V_applied_tensor_accessor[i] = expf(V_applied_tensor_accessor[i]);
      } else {
        V_applied_tensor_accessor[i] = -expf(V_applied_tensor_accessor[i]);
      }
      // delete s;
      // delete x_;
    }



    
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
  solve_sparse_linear(ABCD, E_matrix_host, 2 * m * n);
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
    V_applied_tensor = at::sum(at::mul(V_applied_tensor, conductance_matrix), 0);
    std::cout << V_applied_tensor << std::endl;
    float *V_WL_accessor = V_WL.data_ptr<float>();
    float *V_applied_tensor_accessor = V_applied_tensor.data_ptr<float>();
    int V_applied_tensor_numel = V_applied_tensor.numel();
    // std::cout << V_applied_tensor_numel << std::endl;
    int ADC_resolution = 8; // temp
    float overflow_rate = 0.0f; // temp
    // std::cout << "HERE" << std::endl;
    // float sf = det_sf(V_applied_tensor_accessor, V_applied_tensor_numel, ADC_resolution, overflow_rate, NULL, NULL);
    // std::cout << "HERE" << std::endl;
    // std::cout << sf << std::endl;

    at::Tensor test_tensor = torch::ones(4, torch::kFloat32);
    test_tensor[0] = 0.1f;
    test_tensor[1] = 0.4f;
    test_tensor[2] = 18.0f;
    test_tensor[3] = 2.6f;
    test_tensor = test_tensor.to(torch::Device("cuda:0"));
    float *test_tensor_accessor = test_tensor.data_ptr<float>();
    float *sf;
    cudaMalloc(&sf, sizeof(float));
    det_sf_test<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(test_tensor_accessor, 4, 4, 0.0f, sf);
    cudaSafeCall(cudaDeviceSynchronize());
    quantize_test<<<dim3(1, 1, 1), dim3(4, 1, 1)>>>(test_tensor_accessor, 4, 4, sf);
    cudaSafeCall(cudaDeviceSynchronize());
    std::cout << "HERE" << std::endl;
    std::cout << test_tensor << std::endl;
    std::cout << "HERE" << std::endl;

    // float *sf;
    // cudaMalloc(&sf, sizeof(float));
    // det_sf_test<<<dim3(1, 1, 1), dim3(1, 1, 1)>>>(V_applied_tensor_accessor, V_applied_tensor_numel, ADC_resolution, overflow_rate, sf);
    // cudaSafeCall(cudaDeviceSynchronize());
    // quantize_test<<<dim3(1, 1, 1), dim3(V_applied_tensor_numel, 1, 1)>>>(V_applied_tensor_accessor, V_applied_tensor_numel, ADC_resolution, sf);
    // cudaSafeCall(cudaDeviceSynchronize());
    // std::cout << V_applied_tensor << std::endl;
    return V_applied_tensor;
    // return at::sum(at::mul(V_applied_tensor, conductance_matrix), 0);
  }
}